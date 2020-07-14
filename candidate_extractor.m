function candidate_extractor(indir_path, outdir_path)
indir_path
addpath('./candidate_func');
param.first_frame_no = 1;
param.delayed_frame_no = 20;

% C2C12
threshold_candidate = 30; % intensity threshold for candidate detection
param.minCellArea = 150; % cell size threshold for candidate detection
patch_size = 20; % the size of the patch for candidate detection
scale = 1.0;
threshold_decision = 0.0;

for fidx=1:16
    indir = sprintf([indir_path 'F%04d/'], fidx);
    outpref = 'Cell_seq01_';
    outdir = sprintf([outdir_path 'F%04d/'],fidx);
    mkdir(outdir);

    param.InputImgPost = '.tif';

    param.ResultXmlPost = '.mat';

    % set parameters
    first_frame_no = param.first_frame_no;
    delayed_frame_no = param.delayed_frame_no;

    InputImgPost = param.InputImgPost;
    ResultXmlPost = param.ResultXmlPost;

    %in/output image file directories, prefix and postfix
    InputImgDir = indir;
    ResultXmlDir = outdir;
    mkdir( ResultXmlDir );
    ResultXmlPre = outpref;

    %time interval to wait
    interval = 5; %60 seconds;

    imgprocessed = 0; %img processed
    while 1
        img_files = dir([InputImgDir '*' InputImgPost]);        
        totimgs = length(img_files);            
        if totimgs == imgprocessed
            pause(interval); %if there is no new imgs, wait for a while
            continue;
        end 

        for frame_no = imgprocessed+1:totimgs            

            fprintf( 'frame %d processed\n', frame_no );    

            imagefilepath = sprintf('%s%s',InputImgDir,img_files(frame_no).name);
            resultxmlfilepath = [ResultXmlDir 'candidate'];

            im = imread( imagefilepath );
            im_size = size(im);
            im = padarray(im, [30,30], 'symmetric');

            im = im(:,:,1);

            candidate_sequence_file = resultxmlfilepath;
            candidate_detection_online_optassociate( im, patch_size, threshold_candidate, candidate_sequence_file, first_frame_no, frame_no, delayed_frame_no, param );

        end
        break;

        imgprocessed = totimgs + 1;
    end
end

exit
end