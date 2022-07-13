tic
clc

vidObj = VideoReader('E:\Kampus\Semester 8\Maret\Tes3Org.m4v');
load 'E:\Kampus\Semester 8\Maret\Database_TrainApril' %%memanggil Database yang akan digunakan

%%save video
%v = VideoWriter('hasil1.m4v','MPEG-4');
%v.FrameRate=30;
%open(v);

%%deteksi wajah
faceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP','MergeThreshold', 10); 
tracker = MultiObjectTrackerKLT;
box_color = {'yellow'};

%%Membaca Video
frame = readFrame(vidObj);
frameSize = size(frame);

videoPlayer  = vision.VideoPlayer('Position',[0 0 fliplr(frameSize(1:2))]);

bboxes = [];
tpegawaiA=0;
tpegawaiB=0;
tpegawaiC=0;

frameNumber = 0;
keepRunning = true;
person="";
time=zeros(1,100);
disp('tekan Ctrl-C untuk keluar...');
while hasFrame(vidObj)
    
    framergb = readFrame(vidObj);
    frame = rgb2gray(framergb);
    
   if mod(frameNumber, 10) == 0

       bboxes = []; 
       bboxes = faceDetector.step(frame); %%Wajah terdeteksi pada setiap frame dimunculkan bbox
        if ~isempty(bboxes)
          tracker.addDetections(frame, bboxes);
        end
  else
        tracker.track(frame);
  end
    string="";
    for detek=1:size(tracker.Bboxes,1)
        getimage = imcrop(frame,tracker.Bboxes(detek,:));
        getimage = imresize(getimage, [300 300]);
        queryFeatures = extractHOGFeatures(getimage, 'Cellsize', [8 8]);
        [orang,probs] = predict(faceClassifierModel,queryFeatures); %%SVM          
        orang=cell2mat(orang);
        
        %%Pengenalan Wajah
        prob2 = round(100+100*max(probs));
        
        if prob2 >= 60 %%dimunculkan hasil klasifikasi bbox sesuai hasil skor probabilitas
            switch orang
            case "pegawaiA"
                tpegawaiA=tpegawaiA+1/vidObj.FrameRate;
                lama=strcat(orang," ",num2str(round(tpegawaiA,2)),"s");
            case "pegawaiB"
                tpegawaiB=tpegawaiB+1/vidObj.FrameRate;
                lama=strcat(orang," ",num2str(round(tpegawaiB,2)),"s");
            case "pegawaiC"
                tpegawaiC=tpegawaiC+1/vidObj.FrameRate;
                lama=strcat(orang," ",num2str(round(tpegawaiC,2)),"s");
            end
            string(detek) = strcat(lama," ",num2str(prob2),"%"); %%menampilkan bbox tiap pegawai yg terdeteksi
        else
            string(detek) = strcat("Orang Tidak dikenali"); %%menampilkan bbox orang tidak dikenali jika kondisi yg ditentukan tidak terpenuhi
        end
          
    end
  if ~isempty(tracker.Points)
    displayFrame = insertObjectAnnotation(framergb, 'rectangle',...
        tracker.Bboxes, tracker.BoxIds);
    %displayFrame = insertMarker(displayFrame, tracker.Points); %%Menampilkan tracker points pada wajah
    string = cellstr(string);
    displayFrame = insertText(displayFrame,tracker.Bboxes(:,1:2)-48,string,'FontSize',18); %Menampilkan text pada bbox
   videoPlayer.step(displayFrame);
     
  else
    videoPlayer.step(framergb);

  end
    frameNumber = frameNumber + 1;
    %writeVideo(v,displayFrame);
    %imwrite(displayFrame,sprintf('frame%d.jpg',frameNumber));
end
release(videoPlayer);
%close(v);
toc