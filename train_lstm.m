%% 1. Hiperparametry i Dane
input_size = 4;     % HR, Pace
hidden_size = 128;   % LSTM memory size
epochs = 50;        % Zwiększyłem dla testu
batch_size = 256;
learnRate = 0.001;   % Zmieniona nazwa zmiennej dla clarity

% Generowanie sztucznych danych (jeśli nie masz wczytanych dlX_train)
% Format CBT: [Channels=2, Batch=50, Time=100]
if ~exist('dlX_train', 'var')
    raw_data = randn(input_size, 50, 100); 
    dlX_train = dlarray(raw_data, 'CBT');
    fprintf('Wygenerowano sztuczne dane do testu.\n');
end

%% 2. Inicjalizacja Wag (Ważne: dlarray i struktura)
limit = sqrt(6 / (input_size + hidden_size)); % glorot

params = struct();

% Encoder
params.W_enc = dlarray((rand(4 * hidden_size, hidden_size + input_size)*2 - 1) * limit);
params.b_enc = dlarray(zeros(4 * hidden_size, 1));
params.b_enc(hidden_size+1: 2*hidden_size) = 1; % Forget gate bias trick

% Decoder
params.W_dec = dlarray((rand(4 * hidden_size, hidden_size + input_size)*2 - 1) * limit);
params.b_dec = dlarray(zeros(4 * hidden_size, 1));
params.b_dec(hidden_size+1: 2*hidden_size) = 1;

% Projection
params.W_proj = dlarray((rand(input_size, hidden_size) * 2 - 1) * sqrt(6 / (hidden_size + input_size)));
params.b_proj = dlarray(zeros(input_size, 1));

%% 3. Konfiguracja Treningu
% Zmienne stanu dla optymalizatora Adam
avgG = [];
avgSqG = [];

%% Konfiguracja wykresu
figure;
lineLoss = animatedline('Color', 'b', 'LineWidth', 1.5, 'DisplayName', 'Training Loss');
lineValLoss = animatedline('Color', 'r', 'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', 'Validation Loss');
legend('show');
xlabel('Iteracja'); ylabel('Loss (MSE)');
title('Trening Autoenkodera');
grid on;

numObservations = size(dlX_train, 2); 
iteration = 0;
start = tic;

%% 4. Pętla Treningowa
for epoch = 1:epochs
    % Tasowanie danych treningowych
    idxRandom = randperm(numObservations);
    X_shuffled = dlX_train(:, idxRandom, :);

    % --- PĘTLA PO BATCHACH (TRENING) ---
    for i = 1:batch_size:numObservations
        iteration = iteration + 1;

        idxEnd = min(i + batch_size - 1, numObservations);
        X_batch = X_shuffled(:, i:idxEnd, :);

        % Obliczenie Gradientów i Straty
        [loss, gradients] = dlfeval(@modelGradients, X_batch, params);

        % Aktualizacja Wag (Adam)
        [params, avgG, avgSqG] = adamupdate(params, gradients, ...
            avgG, avgSqG, iteration, learnRate);

        % Wizualizacja Treningu (co iterację - tak jak chciałeś)
        lossValue = double(gather(extractdata(loss)));
        addpoints(lineLoss, iteration, lossValue);
        drawnow limitrate;

        % Raport co batch
        fprintf('Epoka %d | Batch %d/%d | Train Loss: %.6f\n', ...
            epoch, ceil(i/batch_size), ceil(numObservations/batch_size), lossValue);
    end

    % --- WALIDACJA (Raz na epokę) ---
    % Sprawdzamy, czy mamy dane walidacyjne
    if exist('dlX_val', 'var') && ~isempty(dlX_val)

        % 1. Forward Pass na całym zbiorze walidacyjnym
        % (Nie używamy dlfeval, bo nie potrzebujemy gradientów!)
        [~, val_recon] = model_forward(dlX_val, params);

        % 2. Obliczenie Loss (MSE)
        % stripdims jest konieczne, bo dlX_val ma etykiety 'CBT', a val_recon nie
        diff_val = stripdims(dlX_val) - val_recon;
        val_loss = mean(diff_val(:).^2);

        % 3. Ekstrakcja wartości i rysowanie
        valLossValue = double(gather(extractdata(val_loss)));

        % Rysujemy punkt walidacji w bieżącej iteracji
        addpoints(lineValLoss, iteration, valLossValue);
        drawnow;

        % Rozszerzony raport końcowy
        fprintf('>>> EPOKA %d ZAKOŃCZONA. Train Loss: %.6f | VAL LOSS: %.6f <<<\n', ...
            epoch, lossValue, valLossValue);
    else
        fprintf('Epoka %d zakończona. Loss: %.6f (Brak zbioru walidacyjnego)\n', ...
            epoch, lossValue);
    end
end
toc(start);


%% 1. Rekonstrukcja zbioru testowego
% Przepuszczamy CAŁY zbiór testowy naraz (lub w batchach, jeśli jest gigantyczny)
% Funkcja model_forward zwraca zrekonstruowane dane (bez liczenia gradientów)
[~, X_test_recon] = model_forward(dlX_test, params);

%% 2. Obliczenie błędu dla każdego treningu osobno
% X_test_recon jest "czyste" (unformatted), dlX_test ma etykiety 'CBT'
% Musimy zdjąć etykiety do odejmowania
X_test_orig = stripdims(dlX_test);

% Różnica kwadratowa
diff = (X_test_orig - X_test_recon).^2;

% Obliczamy średnią po Cechach (wymiar 1) i Czasie (wymiar 3)
% Zostanie nam wektor o wymiarze [1, LiczbaTreningów]
anomaly_scores = mean(diff, [1 3]);

% Wyciągamy dane z GPU/dlarray do zwykłego wektora liczb
scores = double(gather(extractdata(anomaly_scores)));

% Spłaszczamy do wektora kolumnowego
scores = scores(:);

%% 3. Wizualizacja i Wykrywanie Anomalii
% Ustalmy próg (Threshold). Wszystko powyżej to anomalia.
% Typowa metoda statystyczna: Średnia + 2 lub 3 Odchylenia Standardowe
threshold = mean(scores) + 2 * std(scores);

figure;
% Wykres słupkowy błędów
bar(scores, 'FaceColor', [0.3 0.3 0.3]); hold on;

% Czerwona linia progu
yline(threshold, 'r--', 'LineWidth', 2, 'Label', 'Próg Anomalii');

% Zaznaczamy anomalie na czerwono
anomalies_idx = find(scores > threshold);
bar(anomalies_idx, scores(anomalies_idx), 'FaceColor', 'r');

title('Wykrywanie Anomalii w Danych Testowych');
xlabel('ID Treningu (Testowego)');
ylabel('Błąd Rekonstrukcji (MSE)');
legend('Normalny Trening', 'Próg', 'Anomalia');
grid on;

%% 4. Raport tekstowy
fprintf('\n--- RAPORT ANALIZY ---\n');
fprintf('Liczba treningów testowych: %d\n', length(scores));
fprintf('Wykryto anomalii: %d\n', length(anomalies_idx));

if ~isempty(anomalies_idx)
    fprintf('\nPodejrzane treningi (ID): ');
    fprintf('%d ', anomalies_idx);
    fprintf('\n');

    % Wyświetl szczegóły najgorszego treningu
    [max_score, worst_idx] = max(scores);
    fprintf('Najbardziej "dziwny" trening to ID %d z błędem %.4f\n', worst_idx, max_score);
end


%% WIZUALIZACJA PORÓWNAWCZA (Poprawiona)

% 1. Wybór treningu do wyświetlenia
% Jeśli nie wykryto anomalii, pokażemy pierwszy lepszy trening z brzegu
if exist('anomalies_idx', 'var') && ~isempty(anomalies_idx)
    target_idx = anomalies_idx(1);
    fprintf('Pokazuję anomalię nr indeksu: %d\n', target_idx);
else
    target_idx = 1;
    fprintf('Brak anomalii. Pokazuję trening nr: %d\n', target_idx);
end

% 2. Ekstrakcja danych (Kluczowy moment: SQUEEZE)
% Pobieramy plaster [Cechy x 1 x Czas]
raw_orig = dlX_test(:, target_idx, :);
raw_recon = X_test_recon(:, target_idx, :);

% stripdims -> usuwa etykiety 'CBT'
% extractdata -> wyciąga dane z grafu obliczeniowego
% gather -> ściąga z GPU (jeśli używasz)
% double -> konwertuje na liczby
% squeeze -> zmienia wymiar [2 x 1 x 200] na [2 x 200]
orig_seq = squeeze(double(gather(extractdata(stripdims(raw_orig)))));
recon_seq = squeeze(double(gather(extractdata(stripdims(raw_recon)))));

% Sprawdzenie wymiarów (Dla pewności wypisujemy w konsoli)
fprintf('Wymiary do wykresu: %s\n', mat2str(size(orig_seq))); 
% Powinno być np. [2 100] lub [2 200]

% 3. Rysowanie
t = 1:size(orig_seq, 2); % Oś czasu

figure('Name', ['Analiza Treningu ' num2str(target_idx)], 'Color', 'w');

% Wykres 1: Tętno (Cecha nr 1)
subplot(2,1,1);
plot(t, orig_seq(1,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, recon_seq(1,:), 'r--', 'LineWidth', 1.5);
title('Kanał 1: Tętno (Z-Normalized)');
legend('Oryginał', 'Rekonstrukcja');
grid on;
ylabel('Wartość Z-Score');

% Wykres 2: Tempo (Cecha nr 2)
subplot(2,1,2);
plot(t, orig_seq(2,:), 'g', 'LineWidth', 1.5); hold on;
plot(t, recon_seq(2,:), 'm--', 'LineWidth', 1.5);
title('Kanał 2: Tempo (Z-Normalized)');
legend('Oryginał', 'Rekonstrukcja');
grid on;
xlabel('Czas (znormalizowany)');
ylabel('Wartość Z-Score');


%% WIZUALIZACJA PORÓWNAWCZA - poprawnie odwzorowany trening

% 1. Wybór treningu do wyświetlenia
% Jeśli nie wykryto anomalii, pokażemy pierwszy lepszy trening z brzegu
% 2. Ekstrakcja danych (Kluczowy moment: SQUEEZE)
% Pobieramy plaster [Cechy x 1 x Czas]
[~, X_val_recon] = model_forward(dlX_val, params);

raw_val_orig = dlX_val(:, 5, :);
raw_val_recon = X_val_recon(:, 5, :);

% stripdims -> usuwa etykiety 'CBT'
% extractdata -> wyciąga dane z grafu obliczeniowego
% gather -> ściąga z GPU (jeśli używasz)
% double -> konwertuje na liczby
% squeeze -> zmienia wymiar [2 x 1 x 200] na [2 x 200]
orig_val_seq = squeeze(double(gather(extractdata(stripdims(raw_val_orig)))));
recon_val_seq = squeeze(double(gather(extractdata(stripdims(raw_val_recon)))));

% Sprawdzenie wymiarów (Dla pewności wypisujemy w konsoli)
fprintf('Wymiary do wykresu: %s\n', mat2str(size(orig_val_seq))); 
% Powinno być np. [2 100] lub [2 200]

% 3. Rysowanie
t = 1:size(orig_val_seq, 2); % Oś czasu

figure('Name', ['Analiza Treningu walidacyjnego' num2str(1)], 'Color', 'w');

% Wykres 1: Tętno (Cecha nr 1)
subplot(2,1,1);
plot(t, orig_val_seq(1,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, recon_val_seq(1,:), 'r--', 'LineWidth', 1.5);
title('Kanał 1: Tętno (Z-Normalized)');
legend('Oryginał', 'Rekonstrukcja');
grid on;
ylabel('Wartość Z-Score');

% Wykres 2: Tempo (Cecha nr 2)
subplot(2,1,2);
plot(t, orig_val_seq(2,:), 'g', 'LineWidth', 1.5); hold on;
plot(t, recon_val_seq(2,:), 'm--', 'LineWidth', 1.5);
title('Kanał 2: Tempo (Z-Normalized)');
legend('Oryginał', 'Rekonstrukcja');
grid on;
xlabel('Czas (znormalizowany)');
ylabel('Wartość Z-Score');

%% --- FUNKCJE POMOCNICZE (LOCAL FUNCTIONS) ---

% Ta funkcja spina obliczenia strat z automatycznym różniczkowaniem
function [loss, gradients] = modelGradients(X, params)
    % 1. Forward Pass
    [~, reconstructed_seq] = model_forward(X, params);
    
    % 2. Obliczenie Loss (MSE)
    diff = X - reconstructed_seq;
    loss = mean(diff(:).^2);
    
    % 3. Automatyczne obliczenie gradientów (BPTT robi się samo!)
    gradients = dlgradient(loss, params);
end

% Główna logika przepływu danych
function [loss, reconstructed_seq] = model_forward(input_seq, params)
    % Rozpakowanie parametrów dla czytelności (opcjonalne, ale wygodne)
    W_enc = params.W_enc; b_enc = params.b_enc;
    W_dec = params.W_dec; b_dec = params.b_dec;
    W_proj = params.W_proj; b_proj = params.b_proj;

    % 1. ENKODER
    [h_context, c_context] = run_encoder(input_seq, W_enc, b_enc);
    
    % 2. DEKODER
    seq_len = size(input_seq, 3); % Wymiar 3 to czas w formacie CBT
    reconstructed_seq = run_decoder(h_context, c_context, W_dec, b_dec, W_proj, b_proj, seq_len);
    
    % Loss liczymy w funkcji wyżej (modelGradients), tutaj zwracamy wynik
    loss = 0; 
end

function [h_final, c_final] = run_encoder(input_seq, W_enc, b_enc)
    % Pobieramy wymiary. input_seq jest typu dlarray z etykietami 'CBT'
    % size(input_seq) zwróci [Channel, Batch, Time]
    [~, batch_size, seq_len] = size(input_seq); 
    
    hidden_size = length(b_enc) / 4;
    
    % Inicjalizacja stanów (bez etykiet - unformatted dlarray)
    h_curr = dlarray(zeros(hidden_size, batch_size));
    c_curr = dlarray(zeros(hidden_size, batch_size));
    
    for t = 1:seq_len
        % A. Pobranie próbki danych
        % Oryginalnie: x_t = input_seq(:, :, t); <- To zachowuje etykiety 'CB'
        % POPRAWKA: stripdims usuwa etykiety, umożliwiając mnożenie W * x
        x_t = stripdims(input_seq(:, :, t));
        
        % B. Krok LSTM
        [h_curr, c_curr] = lstm_step(x_t, h_curr, c_curr, W_enc, b_enc);
    end
    
    h_final = h_curr;
    c_final = c_curr;
end


function reconstructed_seq = run_decoder(h_final, c_final, W_dec, b_dec, W_proj, b_proj, seq_len)
    [hidden_size, batch_size] = size(h_final);
    input_dim = size(W_proj, 1);
    
    % Przygotowanie tablicy na wynik (Channel, Batch, Time)
    % Inicjalizujemy jako dlarray, żeby zachować śledzenie gradientów
    reconstructed_seq = dlarray(zeros(input_dim, batch_size, seq_len));
    
    h_curr = h_final;
    c_curr = c_final;
    
    % POPRAWKA: Dummy input musi mieć rozmiar batcha!
    dummy_input = dlarray(zeros(input_dim, batch_size)); 
    
    for t = 1:seq_len
        [h_curr, c_curr] = lstm_step(dummy_input, h_curr, c_curr, W_dec, b_dec);
        
        prediction = W_proj * h_curr + b_proj;
        reconstructed_seq(:, :, t) = prediction;
    end
end

function [h_next, c_next] = lstm_step(x_t, h_prev, c_prev, W, b)
    H = size(h_prev, 1);
    
    % Konkatenacja: [h_prev; x_t]
    z = [h_prev; x_t];
    
    % Wielkie mnożenie
    A = W * z + b;
    
    % Rozdzielenie bramek (Slicing)
    i_gate = sigmoid(A(1:H, :));        
    f_gate = sigmoid(A(H+1:2*H, :));    
    o_gate = sigmoid(A(2*H+1:3*H, :));  
    g_gate = tanh(A(3*H+1:4*H, :));     
    
    % Update stanów
    c_next = (f_gate .* c_prev) + (i_gate .* g_gate);
    h_next = o_gate .* tanh(c_next);
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end