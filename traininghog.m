clc
clear all

%% Memanggil folder yang akan dijadikan Database
faceDatabase = imageSet('E:\Kampus\Semester 8\Maret\FIX\database_train1','recursive');
%faceDatabase = imageSet('E:\Kampus\Semester 8\Maret\FIX\database_train','recursive');

%% Membagi Data Menjadi DataTrain dan DataTest
[training,test] = partition(faceDatabase,[0.8 0.2]);

%% Extract and display Histogram of Oriented Gradient Features
person = 3;
[hogFeature, visualization]= ...
    extractHOGFeatures(read(training(person),1));

% Ekstraksi Fitur HOG untuk Training
trainingFeatures = zeros(size(training,2)*training(1).Count,46656);
featureCount = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
        points = detectSURFFeatures(read(training(i),j));
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
        trainingLabel{featureCount} = training(i).Description;    
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end
trainingFeature = trainingFeatures(1:size(trainingLabel,2),:);
%t = templateSVM('KernelFunction','polynomial','PolynomialOrder',2);
faceClassifierModel = fitcecoc(trainingFeature,trainingLabel, 'coding', 'onevsall', 'FitPosterior',1);
[prediksi,~,~,Posterior] = predict(faceClassifierModel, trainingFeature);

%save ..\Maret\Database_Train
save('Database_TrainApril.mat','faceClassifierModel');