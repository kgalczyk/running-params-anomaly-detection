% Wymagane zmienne w Workspace: params, mu, sig (te z treningu!)
winSize = 64; 
stride = 16; 

file = 'output_all/.csv';

% Testujemy plik z anomaliami
analyze_full_session(file, params, mu, sig, winSize, stride);
visualize_brain(file, params, mu, sig, winSize, stride);

function analyze_full_session(filename, params, mu, sig, windowSize, stride)
    % 1. Wczytanie i przygotowanie pełnego pliku
    opts = detectImportOptions(filename);
    opts.VariableNamingRule = 'preserve';
    tbl = readtable(filename, opts);
    rawData = table2array(tbl);
    rawData(isnan(rawData)) = 0;
    
    fullLen = size(rawData, 1);
    num_features = 2; % HR, Pace
    
    % Normalizacja całego pliku (tymi samymi statystykami co trening!)
    mu_vec = squeeze(mu); 
    sig_vec = squeeze(sig);
    % Transpozycja do [Features x Time] dla łatwiejszych obliczeń
    normDataFull = ((rawData(:, 1:num_features)' - mu_vec) ./ sig_vec);
    
    % 2. Przygotowanie buforów na rekonstrukcję
    % Będziemy sumować wyniki z nakładających się okien
    reconAccumulator = zeros(size(normDataFull));
    overlapCounter = zeros(1, fullLen);
    
    % 3. PĘTLA SKANUJĄCA (Inference)
    num_windows = floor((fullLen - windowSize) / stride) + 1;
    
    fprintf('Analiza pliku: %s\n', filename);
    fprintf('Liczba okien do przetworzenia: %d\n', num_windows);
    
    for w = 0:(num_windows-1)
        start_idx = w * stride + 1;
        end_idx = start_idx + windowSize - 1;
        idx_range = start_idx:end_idx;
        
        % Wycinek [Features x Time]
        chunk = normDataFull(:, idx_range);
        
        % Konwersja na dlarray [Channel, Batch=1, Time]
        dlChunk = dlarray(reshape(chunk, [num_features, 1, windowSize]), 'CBT');
        
        % Predykcja (Forward Pass)
        [~, dlRecon] = model_forward(dlChunk, params);
        
        % Wyciągnięcie danych
        chunkRecon = double(gather(extractdata(stripdims(dlRecon))));
        chunkRecon = squeeze(chunkRecon); % -> [Features x WindowSize]
        
        % AKUMULACJA (Dodajemy wynik do bufora w odpowiednim miejscu)
        reconAccumulator(:, idx_range) = reconAccumulator(:, idx_range) + chunkRecon;
        overlapCounter(idx_range) = overlapCounter(idx_range) + 1;
    end
    
    % 4. UŚREDNIANIE (Dzielimy przez liczbę nakładek)
    % Unikamy dzielenia przez zero na krawędziach (tam gdzie okno nie doszło)
    valid_mask = overlapCounter > 0;
    finalRecon = zeros(size(reconAccumulator));
    finalRecon(:, valid_mask) = reconAccumulator(:, valid_mask) ./ overlapCounter(valid_mask);
    
    % 5. OBLICZENIE BŁĘDU (Anomaly Score)
    % Błąd punktowy dla każdego momentu czasu
    % (Oryginał - Rekonstrukcja)^2
    diffSq = (normDataFull - finalRecon).^2;
    anomalyScore = sum(diffSq, 1); % Suma błędów HR i Pace w danym momencie
    
    % Wygładzenie wyniku błędu (żeby wykres był czytelniejszy)
    anomalyScoreSmooth = movmean(anomalyScore, 20);
    
    % --- WIZUALIZACJA ---
    t = 1:fullLen;
    
    figure('Name', ['Analiza Pełna: ' filename], 'Color', 'w', 'Position', [100 100 1200 800]);
    
    % Wykres 1: Tętno
    subplot(3,1,1);
    plot(t, normDataFull(1,:), 'b', 'LineWidth', 1.5); hold on;
    plot(t, finalRecon(1,:), 'r', 'LineWidth', 1.5);
    title('Tętno (Oryginał vs Rekonstrukcja)');
    legend('Oryginał', 'Model');
    grid on; xlim([1 fullLen]);
    
    % Wykres 2: Tempo
    subplot(3,1,2);
    plot(t, normDataFull(2,:), 'g', 'LineWidth', 1.5); hold on;
    plot(t, finalRecon(2,:), 'm', 'LineWidth', 1.5);
    title('Tempo (Oryginał vs Rekonstrukcja)');
    legend('Oryginał', 'Model');
    grid on; xlim([1 fullLen]);
    
    % Wykres 3: DETEKTOR ANOMALII
    subplot(3,1,3);
    area(t, anomalyScoreSmooth, 'FaceColor', [1 0.7 0.7], 'EdgeColor', 'r');
    title('Wskaźnik Anomalii (Im wyżej, tym gorzej)');
    xlabel('Czas (próbki)');
    ylabel('MSE');
    grid on; xlim([1 fullLen]);
    
    % Rysowanie progu alarmowego (np. 3 sigma błędu)
    threshold = mean(anomalyScoreSmooth) + 3*std(anomalyScoreSmooth);
    yline(threshold, 'k--', 'Próg Anomalii', 'LineWidth', 2);
end

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

function visualize_brain(filename, params, mu, sig, windowSize, stride)
    % Używamy tej samej logiki co przy analizie, ale patrzymy na h_final
    opts = detectImportOptions(filename); opts.VariableNamingRule = 'preserve';
    tbl = readtable(filename, opts); raw = table2array(tbl); raw(isnan(raw))=0;
    
    % Normalizacja
    mu_vec = squeeze(mu); sig_vec = squeeze(sig);
    normData = ((raw(:, 1:2)' - mu_vec) ./ sig_vec);
    
    [~, fullLen] = size(normData);
    num_windows = floor((fullLen - windowSize) / stride) + 1;
    
    % Macierz na historię stanów ukrytych [HiddenSize x LiczbaOkien]
    hidden_size = size(params.b_enc, 1) / 4;
    latent_history = zeros(hidden_size, num_windows);
    
    fprintf('Skanowanie aktywności neuronów...\n');
    
    for w = 0:(num_windows-1)
        start_idx = w * stride + 1;
        chunk = normData(:, start_idx : start_idx+windowSize-1);
        dlChunk = dlarray(reshape(chunk, [2, 1, windowSize]), 'CBT');
        
        % Uruchamiamy TYLKO ENKODER
        % Musisz mieć funkcję run_encoder dostępną w ścieżce
        [h, ~] = run_encoder(dlChunk, params.W_enc, params.b_enc);
        
        latent_history(:, w+1) = double(gather(extractdata(h)));
    end
    
    % Rysowanie
    figure('Color', 'w', 'Name', 'Aktywność Enkodera');
    
    subplot(2,1,1);
    plot(normData(1,:), 'r'); hold on; plot(normData(2,:), 'b');
    legend('Tętno', 'Tempo'); title('Twoje Dane'); axis tight;
    
    subplot(2,1,2);
    % Heatmapa aktywności neuronów
    imagesc(latent_history); 
    colormap('jet'); colorbar;
    xlabel('Czas (kolejne okna)'); ylabel('Neuron nr...');
    title('Co "widzi" sieć? (Latent Space Activation)');
end