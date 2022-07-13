clear all
clc

vidObj = VideoReader('E:\Kampus\Semester 8\Maret\vidtrain\pegawaiB.m4v');
faceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP'); 
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

videoFrame = readFrame(vidObj);
frameSize = size(videoFrame);
numPts = 0;
frameCount = 1;

while hasFrame(vidObj)
    videoFrame = readFrame(vidObj);
    videoFrameGray = rgb2gray(videoFrame);
    %C = videoFrameGray*1; %Menambah gelap framegrayscale
    frameCount = frameCount + 1;
    if numPts < 10
        bbox = faceDetector.step(videoFrameGray);
        if ~isempty(bbox)
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);
            oldPoints = xyPoints;
            bboxPoints = bbox2points(bbox(1, :));
            bboxPolygon = reshape(bboxPoints', 1, []);
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end

    else
        
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);

        numPts = size(visiblePoints, 1);

        if numPts >= 10
  
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

            bboxPoints = transformPointsForward(xform, bboxPoints);

            bboxPolygon = reshape(bboxPoints', 1, []);
            
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');

            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end

    end
    position1 = min(bboxPolygon(2),bboxPolygon(4));
    position2 = max(bboxPolygon(6),bboxPolygon(8));
    position3 = min(bboxPolygon(1),bboxPolygon(7));
    position4 = max(bboxPolygon(3),bboxPolygon(5));
    
    getimage = videoFrameGray(position1:position2,position3:position4,:);
    getimage = imresize(getimage, [300 300]);
    imwrite(getimage,sprintf('w%04d.JPG',frameCount));
end