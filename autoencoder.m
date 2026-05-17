% hiperparams

input_size = 2;     % HR, Pace
hidden_size = 16;   % LSTM memory size
latent_size = 8;    % compressed training 'image'

epochs = 10;
batch_size = 10;
alfa = 0.01;

% wages init
limit = sqrt(6 / (input_size + hidden_size)); % glorot

% encoder wages
W_enc = (rand(4 * hidden_size, hidden_size + input_size)*2 - 1) *limit;
b_enc = zeros(4 * hidden_size, 1);
b_enc(hidden_size+1: 2*hidden_size) = 1;

% decoder wages
W_dec = (rand(4 * hidden_size, hidden_size + input_size)*2 - 1) *limit;
b_dec = zeros(4 * hidden_size, 1);
b_dec(hidden_size+1: 2*hidden_size) = 1;

% projection wages
W_proj = (rand(input_size, hidden_size) * 2 - 1) * sqrt(6 / (hidden_size + input_size));
b_proj = zeros(input_size, 1);

h_t = zeros(hidden_size, 1);
c_t = zeros(hidden_size, 1);


% Wykres na żywo
figure;
lineLoss = animatedline('Color', 'b');
xlabel('Iteracja'); ylabel('Loss (MSE)');
title('Trening Autokodera');
grid on;

numObservations = size(dlX_train, 2); % Wymiar Batch to 2 (bo mamy CBT)
iteration = 0;
start = tic;

for epoch = 1:epochs
    % 1. Tasowanie danych na początku epoki
    idxRandom = randperm(numObservations);
    X_shuffled = dlX_train(:, idxRandom, :);
    
    % 2. Pętla po Batchach
    for i = 1:batch_size:numObservations
        iteration = iteration + 1;
        
        % Wycięcie Batcha (Slicing)
        idxEnd = min(i + batch_size - 1, numObservations);
        X_batch = X_shuffled(:, i:idxEnd, :);
        
        % 3. Obliczenie Gradientów i Straty (To jest klucz!)
        % dlfeval wywołuje naszą funkcję modelGradients i śledzi operacje
        [loss, gradients] = dlfeval(@model_forward, X_batch, params);
        
        % 4. Aktualizacja Wag (Optimizer Adam)
        [params, avgG, avgSqG] = adamupdate(params, gradients, ...
            avgG, avgSqG, iteration, learnRate);
        
        % 5. Wizualizacja (co 10 iteracji)
        if mod(iteration, 10) == 0
            lossValue = double(gather(extractdata(loss)));
            addpoints(lineLoss, iteration, lossValue);
            drawnow limitrate;
        end
    end
    
    % Raport po epoce
    fprintf('Epoka %d/%d zakończona. Ostatni Loss: %.4f\n', ...
        epoch, epochs, double(gather(extractdata(loss))));
end

toc(start);


function [loss, reconstructed_seq] = model_forward(input_seq, W_enc, b_enc, W_dec, b_dec, W_proj, b_proj)
    % 1. KODER: Kompresujemy dane wejściowe do wektora stanu
    [h_context, c_context] = run_encoder(input_seq, W_enc, b_enc);
    
    % 2. DEKODER: Próbujemy odtworzyć dane z wektora stanu
    % Przekazujemy długość sekwencji taką samą jak wejścia
    seq_len = size(input_seq, 2);
    reconstructed_seq = run_decoder(h_context, c_context, W_dec, b_dec, W_proj, b_proj, seq_len);
    
    % 3. LOSS (MSE): Obliczamy błąd rekonstrukcji
    % Loss = średnia z kwadratów różnic między oryginałem a rekonstrukcją
    diff = input_seq - reconstructed_seq;
    loss = mean(diff(:).^2);
end

function [h_next, c_next] = lstm_step(x_t, h_prev, c_prev, W, b)
    H = size(h_prev, 1);

    z = [h_prev; x_t];
    A = W * z + b;

    idx = 1:H;

    i_gate = sigmoid(A(idx)); %Input 

    idx = idx + H;
    f_gate = sigmoid(A(idx)); %Forget

    idx = idx + H;
    o_gate = sigmoid(A(idx)); %Output

    idx = idx + H;
    g_gate = tanh(A(idx));    %Cell(Gate)

    c_next = (f_gate .* c_prev) + (i_gate .* g_gate);
    h_next = o_gate .* tanh(c_next);
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
        x_t = stripdims(input_seq(:, :, t));
        
        % B. Krok LSTM
        [h_curr, c_curr] = lstm_step(x_t, h_curr, c_curr, W_enc, b_enc);
    end
    
    h_final = h_curr;
    c_final = c_curr;
end

function reconstructed_seq = run_decoder(h_final, c_final, W_dec, b_dec, W_proj, b_proj, seq_len)
    N = 2;

    reconstructed_seq = zeros(N, seq_len);

    h_curr = h_final;
    c_curr = c_final;

    dummy_input = zeros(N, 1); % decoder "narysuje" trening znowu

    for t = 1:seq_len
        % A. Krok LSTM (używamy tej samej funkcji co w Enkoderze!)
        [h_curr, c_curr] = lstm_step(dummy_input, h_curr, c_curr, W_dec, b_dec);
        
        % B. Projekcja (Odzyskanie wymiarów danych)
        % Wzór: y = W * h + b
        prediction = W_proj * h_curr + b_proj;
        
        % C. Zapisanie wyniku
        reconstructed_seq(:, t) = prediction;
        
        % (Opcjonalnie: Tutaj w zaawansowanym modelu mógłbyś przypisać 
        % dummy_input = prediction, żeby karmić sieć jej własnym wyjściem)
    end

end

function [dW, db, dx] = lstm_backward_manual(dh_sequence, cache, W)
    % dh_sequence: Błędy otrzymane z wyższych warstw (lub z dekodera) dla każdego czasu t
    % cache: Struktura zawierająca wartości bramek (i, f, o, g) i stanów (c, h) z forward pass
    
    [H, T] = size(dh_sequence); % H - rozmiar ukryty, T - czas
    input_dim = size(W, 2) - H; % Obliczenie rozmiaru wejścia na podstawie wag
    
    % Inicjalizacja gradientów wag (akumulatory)
    dW = zeros(size(W));
    db = zeros(size(W, 1), 1);
    dx = zeros(input_dim, T);
    
    % Inicjalizacja błędów "z przyszłości" (dla ostatniego kroku są zerami)
    dh_next = zeros(H, 1);
    dc_next = zeros(H, 1);
    
    % Backprop over time!!
    for t = T:-1:1
        
        % 1. Pobieramy wartości bramek z cache dla chwili t
        i = cache.i(:, t); f = cache.f(:, t); 
        o = cache.o(:, t); g = cache.g(:, t);
        c = cache.c(:, t); c_prev = cache.c_prev(:, t);
        x = cache.x(:, t); h_prev = cache.h_prev(:, t);
        
        % 2. Całkowity błąd stanu ukrytego h
        % Suma błędu z "góry" (dh_sequence) i błędu z "przyszłości" (dh_next)
        dh = dh_sequence(:, t) + dh_next;
        
        % 3. Pochodna przez bramkę wyjściową (Output Gate)
        do = dh .* tanh(c);
        da_o = do .* i .* (1 - i); % Pochodna sigmoidy: x * (1-x)
        
        % 4. Błąd stanu komórki (Cell State)
        dc = (dh .* o .* (1 - tanh(c).^2)) + dc_next;
        
        % 5. Pochodna przez bramkę zapominania (Forget Gate)
        df = dc .* c_prev;
        da_f = df .* f .* (1 - f); % Pochodna sigmoidy
        
        % 6. Pochodna przez bramkę wejściową (Input Gate)
        di = dc .* g;
        da_i = di .* i .* (1 - i); % Pochodna sigmoidy
        
        % 7. Pochodna przez kandydata (Gate / Update)
        dg = dc .* i;
        da_g = dg .* (1 - g.^2);   % Pochodna tanh: 1 - x^2
        
        % 8. Sklejenie gradientów bramek (z powrotem do wielkiego wektora Z)
        d_gates = [da_i; da_f; da_o; da_g]; % Zakładając kolejność IFOG
        
        % 9. AKUMULACJA GRADIENTÓW WAG (Shared Weights)
        % To jest ten moment, gdzie Backprop zmienia się w BPTT
        z_input = [h_prev; x]; % To co wchodziło do mnożenia wag
        dW = dW + d_gates * z_input';
        db = db + sum(d_gates, 2);
        
        % 10. Obliczenie błędów do przekazania do przeszłości (t-1)
        dz = W' * d_gates; % Wsteczna propagacja przez wagi
        
        % Rozdzielenie na błąd h_prev i x
        dh_next = dz(1:H, :);       % To pójdzie do pętli w następnym obiegu (t-1)
        dx(:, t) = dz(H+1:end, :);  % To jest gradient dla wejścia (dla poprzednich warstw)
        
        % Obliczenie błędu dc_next dla następnego kroku
        dc_next = dc .* f;
    end
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end