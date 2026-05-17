clear all;

%% 1. KONFIGURACJA
normalDir = 'output_all';
anomalyDir = 'output_rejected'; 

% --- ZMIANA: Parametry Okna ---
windowSize = 64;   % Długość jednego wycinka (np. 64 próbki to ok. 1-2 minuty biegu)
stride = 32;       % Przesunięcie (Overlap). Im mniejsze, tym więcej danych.
num_features = 4; 

%% 2. FUNKCJA WCZYTUJĄCA I TNĄCA
% Definiujemy funkcję pomocniczą (zapisz ją na końcu skryptu lub w osobnym pliku)
function [X_chunks] = load_and_slice(directory, num_features, winSize, stride)
    files = dir(fullfile(directory, '*.csv'));
    numFiles = length(files);
    X_chunks = [];
    
    % Konfiguracja pauz i wygładzania
    PAUSE_THRESHOLD = 10; % Sekundy
    SMOOTH_WINDOW = 30;   % Ile próbek uśredniać dla wysokości (10s to dobry standard)
    
    if numFiles == 0, return; end
    fprintf('Przetwarzanie z wykrywaniem pauz i obliczaniem SLOPE...\n');
    
    for i = 1:numFiles
        filePath = fullfile(directory, files(i).name);
        opts = detectImportOptions(filePath);
        opts.VariableNamingRule = 'preserve'; 
        tbl = readtable(filePath, opts);
        
        % 1. Obsługa Timestamp (do cięcia pauz)
        if ~ismember('timestamp', tbl.Properties.VariableNames)
             % Fallback, jeśli brak timestampu
            time_sec = (1:height(tbl))'; 
        else
            if isdatetime(tbl.timestamp)
                time_sec = seconds(tbl.timestamp - tbl.timestamp(1));
            elseif iscell(tbl.timestamp) || isstring(tbl.timestamp)
                dt = datetime(tbl.timestamp, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
                time_sec = seconds(dt - dt(1));
            else
                time_sec = double(tbl.timestamp);
            end
        end
        
        % 2. --- NOWOŚĆ: OBLICZANIE SLOPE (NACHYLENIA) ---
        if ismember('altitude', tbl.Properties.VariableNames)
            raw_alt = tbl.altitude;
            
            % KROK A: Wygładzanie (Moving Average)
            % Surowy GPS skacze, wygładzenie 10-sekundowe daje realny profil terenu
            alt_smooth = movmean(raw_alt, SMOOTH_WINDOW);
            
            % KROK B: Obliczenie pochodnej (Gradient)
            % Zwraca zmianę wysokości na jedną próbkę (m/s w pionie)
            % Używamy gradient() zamiast diff(), żeby zachować długość wektora
            slope = gradient(alt_smooth);
            
            % Opcjonalnie: Wzmocnienie sygnału (np. x10), żeby liczby nie były za małe
            % Ale Z-Score normalization (które masz później) i tak to załatwi.
        else
            warning('Brak kolumny altitude w %s. Wstawiam zera.', files(i).name);
            slope = zeros(height(tbl), 1);
        end
        
        % 3. Składanie macierzy cech [HR, Speed, Cadence, SLOPE]
        try
            % Upewnij się, że kolumny są w tej kolejności!
            dataRaw = [tbl.heart_rate, tbl.speed_ms, tbl.cadence, slope];
        catch
            % Jeśli nazwy są inne, spróbuj po indeksach (mniej bezpieczne)
            dataRaw = [table2array(tbl(:, 2:4)), slope];
        end
        
        dataRaw(isnan(dataRaw)) = 0;
        
        % 4. Cięcie na segmenty (tak jak wcześniej)
        dt = diff(time_sec);
        break_indices = find(dt > PAUSE_THRESHOLD);
        
        segment_starts = [1; break_indices + 1];
        segment_ends = [break_indices; length(time_sec)];
        
        for s = 1:length(segment_starts)
            idx_start = segment_starts(s);
            idx_end = segment_ends(s);
            
            segment_data = dataRaw(idx_start:idx_end, :);
            
            if size(segment_data, 1) < winSize
                continue; 
            end
            
            num_windows = floor((size(segment_data, 1) - winSize) / stride) + 1;
            
            for w = 0:(num_windows-1)
                ws = w * stride + 1;
                we = ws + winSize - 1;
                chunk = segment_data(ws:we, :);
                
                % Permute [Win, Feat] -> [1, Feat, Win]
                chunk_perm = permute(chunk, [3, 2, 1]);
                
                if isempty(X_chunks)
                    X_chunks = chunk_perm;
                else
                    X_chunks = cat(1, X_chunks, chunk_perm);
                end
            end
        end
    end
end

%% 3. WYKONANIE
fprintf('Generowanie okien treningowych...\n');
X_normal_raw = load_and_slice(normalDir, num_features, windowSize, stride);

fprintf('Generowanie okien anomalii...\n');
if exist(anomalyDir, 'dir')
    X_anomaly_raw = load_and_slice(anomalyDir, num_features, windowSize, stride);
else
    X_anomaly_raw = [];
end

% Sprawdzenie ilości danych
numSamples = size(X_normal_raw, 1);
fprintf('--- RAPORT DANYCH ---\n');
fprintf('Zamiast 40 plików, mamy teraz: %d wycinków treningowych!\n', numSamples);

if numSamples < 100
    error('Nadal za mało danych. Zmniejsz "stride" (przesunięcie)!');
end

%% 4. PERMUTACJA DO FORMATU CBT (Channel, Batch, Time)
% Obecnie mamy [Batch, Channel, Time]. Musimy zamienić Batch i Channel.
X_normal = permute(X_normal_raw, [2, 1, 3]); 

if ~isempty(X_anomaly_raw)
    X_anomaly = permute(X_anomaly_raw, [2, 1, 3]);
else
    X_anomaly = [];
end

%% 5. PODZIAŁ I NORMALIZACJA (Standardowa procedura)
rng(42);
idxRandom = randperm(numSamples);

trainRatio = 0.80; 
nTrain = floor(trainRatio * numSamples);

idxTrain = idxRandom(1:nTrain);
idxVal   = idxRandom(nTrain+1:end);

X_train_raw = X_normal(:, idxTrain, :);
X_val_raw   = X_normal(:, idxVal, :);
X_test_raw  = X_anomaly; 

% Normalizacja (Statystyki tylko z treningu!)
mu = mean(X_train_raw, [2 3]); 
sig = std(X_train_raw, 0, [2 3]);
sig(sig < 1e-6) = 1;

X_train = (X_train_raw - mu) ./ sig;
X_val   = (X_val_raw   - mu) ./ sig;
if ~isempty(X_test_raw)
    X_test = (X_test_raw - mu) ./ sig;
else
    X_test = [];
end

% Konwersja na dlarray
dlX_train = dlarray(X_train, 'CBT');
dlX_val   = dlarray(X_val,   'CBT');
if ~isempty(X_test)
    dlX_test = dlarray(X_test, 'CBT');
end

disp('Gotowe do treningu.');

% %% 1. KONFIGURACJA
% normalDir = 'output';           % Folder ze "zdrowymi" treningami
% anomalyDir = 'output_rejected'; % Folder z anomaliami (do testów)
% targetLen = 200;
% num_features = 2; 
% 
% %% 2. WCZYTYWANIE DANYCH (Osobno normalne, osobno anomalie)
% fprintf('--- Wczytywanie zbioru NORMALNEGO (Trening) ---\n');
% [X_normal_raw, names_normal] = load_folder(normalDir, num_features, targetLen);
% 
% fprintf('--- Wczytywanie zbioru ANOMALII (Test) ---\n');
% % Sprawdzamy czy folder istnieje, żeby uniknąć błędu
% if exist(anomalyDir, 'dir')
%     [X_anomaly_raw, names_anomaly] = load_folder(anomalyDir, num_features, targetLen);
% else
%     warning('Folder z anomaliami nie istnieje! Zbiór testowy będzie pusty.');
%     X_anomaly_raw = [];
% end
% 
% %% 3. PRZYGOTOWANIE TENSORÓW (Format CBT)
% % X_normal_raw ma wymiar [Batch, Features, Time]. Musimy zrobić permutację.
% if ~isempty(X_normal_raw)
%     X_normal = permute(X_normal_raw, [2, 1, 3]); % -> [Features, Batch, Time]
% else
%     error('Brak danych treningowych!');
% end
% 
% if ~isempty(X_anomaly_raw)
%     X_anomaly = permute(X_anomaly_raw(10:20,:,:), [2, 1, 3]);
% else
%     X_anomaly = [];
% end
% 
% %% 4. PODZIAŁ DANYCH
% % Strategia: 
% % - Zbiór Normalny dzielimy na Trening (90%) i Walidację (10%).
% % - Zbiór Anomalii idzie w 100% do Testu.
% 
% [numChannels, numNormal, seqLen] = size(X_normal);
% rng(20); % Ziarno losowości
% idxRandom = randperm(numNormal);
% 
% % Proporcje dla zbioru normalnego
% trainRatio = 0.90; 
% nTrain = floor(trainRatio * numNormal);
% 
% % Indeksy
% idxTrain = idxRandom(1:nTrain);
% idxVal   = idxRandom(nTrain+1:end);
% 
% % Przypisanie
% X_train_raw = X_normal(:, idxTrain, :);
% X_val_raw   = X_normal(:, idxVal, :);
% 
% % Zbiór testowy to same anomalie (możesz tu też dodać część normalnych dla porównania)
% X_test_raw  = X_anomaly; 
% 
% fprintf('\nPodział danych:\n');
% fprintf('  Treningowe (Normalne):  %d\n', size(X_train_raw, 2));
% fprintf('  Walidacyjne (Normalne): %d\n', size(X_val_raw, 2));
% fprintf('  Testowe (Anomalie):     %d\n', size(X_test_raw, 2));
% 
% %% 5. NORMALIZACJA (Kluczowy moment)
% % Statystyki liczymy TYLKO na zbiorze TRENINGOWYM (zdrowym).
% % Anomalie normalizujemy według "zdrowych standardów" - to uwypukli ich dziwność.
% 
% mu = mean(X_train_raw, [2 3]); 
% sig = std(X_train_raw, 0, [2 3]);
% sig(sig < 1e-6) = 1; % Zabezpieczenie
% 
% % Aplikacja normalizacji
% X_train = (X_train_raw - mu) ./ sig;
% X_val   = (X_val_raw   - mu) ./ sig;
% 
% if ~isempty(X_test_raw)
%     X_test = (X_test_raw - mu) ./ sig;
% else
%     X_test = [];
% end
% 
% %% 6. KONWERSJA NA DLARRAY
% dlX_train = dlarray(X_train, 'CBT');
% dlX_val   = dlarray(X_val,   'CBT');
% 
% if ~isempty(X_test)
%     dlX_test = dlarray(X_test, 'CBT');
% else
%     dlX_test = dlarray([]);
% end
% 
% disp('Gotowe. Utworzono zmienne: dlX_train, dlX_val, dlX_test (anomalie).');
% 
% 
% %% --- FUNKCJA POMOCNICZA ---
% function [allData, fileNames] = load_folder(directory, num_features, targetLen)
%     files = dir(fullfile(directory, '*.csv'));
%     numFiles = length(files);
% 
%     if numFiles == 0
%         allData = [];
%         fileNames = {};
%         fprintf('  -> Brak plików w folderze: %s\n', directory);
%         return;
%     end
% 
%     allData = zeros(numFiles, num_features, targetLen);
%     fileNames = cell(numFiles, 1);
% 
%     for i = 1:numFiles
%         filePath = fullfile(directory, files(i).name);
%         opts = detectImportOptions(filePath);
%         % Uciszamy ostrzeżenia przy czytaniu
%         opts.VariableNamingRule = 'preserve'; 
%         tbl = readtable(filePath, opts);
% 
%         rawData = table2array(tbl);
%         rawData(isnan(rawData)) = 0;
% 
%         % Interpolacja
%         origLen = size(rawData, 1);
%         x_orig = linspace(0, 1, origLen);
%         x_new = linspace(0, 1, targetLen);
% 
%         resampled = zeros(num_features, targetLen);
%         for feat = 1:num_features
%             % Używamy min, żeby nie wyjść poza zakres, jeśli plik ma mniej kolumn
%             col_idx = min(feat, size(rawData, 2));
%             resampled(feat, :) = interp1(x_orig, rawData(:, col_idx), x_new, 'linear');
%         end
% 
%         allData(i, :, :) = resampled;
%         fileNames{i} = files(i).name;
%     end
%     fprintf('  -> Wczytano %d plików.\n', numFiles);
% end

% % Konfiguracja
% inputDir = 'output'; 
% targetLen = 300; % Docelowa liczba punktów dla każdego treningu
% num_features = 2;
% files = dir(fullfile(inputDir, '*.csv'));
% numFiles = length(files),
% 
% % Inicjalizacja macierzy (Treningi x Cechy x Czas)
% % Kolejność kolumn w CSV: heart_rate, pace, cadence (3 cechy)
% allData = zeros(numFiles, num_features, targetLen); 
% sessionNames = cell(numFiles, 1);
% 
% fprintf('Rozpoczynam przetwarzanie %d plików...\n', numFiles);
% 
% for i = 1:numFiles
%     filePath = fullfile(inputDir, files(i).name);
%     opts = detectImportOptions(filePath);
%     tbl = readtable(filePath, opts);
% 
%     % Konwersja na macierz i usunięcie ewentualnych braków (NaN)
%     rawData = table2array(tbl);
%     rawData(isnan(rawData)) = 0;
% 
%     % Oryginalna oś czasu (próbki)
%     origLen = size(rawData, 1);
%     x_orig = linspace(0, 1, origLen);
%     x_new = linspace(0, 1, targetLen);
% 
%     % Interpolacja każdej cechy z osobna
%     resampled = zeros(num_features, targetLen);
%     for feat = 1:num_features
%         resampled(feat, :) = interp1(x_orig, rawData(:, feat), x_new, 'linear');
%     end
% 
%     % Normalizacja Z-Score (opcjonalna tutaj, ale zalecana przed ML)
%     % Możesz też znormalizować cały tensor po wczytaniu wszystkich plików
%     allData(i, :, :) = resampled;
%     sessionNames{i} = files(i).name;
% end
% 
% % Zamiana na format cell array - wymagany przez MATLABowe warstwy LSTM
% % Każda komórka to [3 x 200] (Cechy x Czas)
% X = squeeze(num2cell(allData, [2, 3]));
% 
% X_permuted = permute(allData, [2, 1, 3]);
% [numChannels, numObs, seqLen] = size(X_permuted);
% 
% fprintf('Dane gotowe. Rozmiar tensora: %s\n', mat2str(size(allData)));
% 
% % Ustawienie proporcji podziału
% trainRatio = 1.0;
% valRatio   = 0.0;
% testRatio  = 0.0;
% 
% rng(41); % Ustawienie ziarna losowości dla powtarzalności wyników
% idxRandom = randperm(numObs);
% X_shuffled = X_permuted(:, idxRandom, :);
% 
% nTrain = floor(trainRatio * numObs);
% nVal   = floor(valRatio * numObs);
% nTest  = numObs - nTrain - nVal; % Reszta idzie do testowego
% 
% fprintf('Podział danych:\n');
% fprintf('  Treningowe:  %d\n', nTrain);
% fprintf('  Walidacyjne: %d\n', nVal);
% fprintf('  Testowe:     %d\n', nTest);
% 
% % Wycinamy odpowiednie fragmenty macierzy
% X_train_raw = X_shuffled(:, 1:nTrain, :);
% X_val_raw   = X_shuffled(:, nTrain+1 : nTrain+nVal, :);
% X_test_raw  = X_shuffled(:, nTrain+nVal+1 : end, :);
% 
% % normalizacja: Obliczamy statystyki wzdłuż wymiarów Batch(2) i Time(3) dla każdego Kanału(1)
% mu = mean(X_train_raw, [2 3]); 
% sig = std(X_train_raw, 0, [2 3]);
% 
% % Zabezpieczenie na wypadek stałej wartości (dzielenie przez zero)
% sig(sig < 1e-6) = 1;
% 
% % Aplikacja normalizacji: (Wartość - Średnia) / Odchylenie
% X_train = (X_train_raw - mu) ./ sig;
% X_val   = (X_val_raw   - mu) ./ sig;
% X_test  = (X_test_raw  - mu) ./ sig;
% 
% % Dodajemy etykietę formatu 'CBT' (Channel, Batch, Time)
% dlX_train = dlarray(X_train, 'CBT');
% dlX_val   = dlarray(X_val,   'CBT');
% dlX_test  = dlarray(X_test,  'CBT');
% 
% disp('Gotowe. Utworzono zmienne: dlX_train, dlX_val, dlX_test.');