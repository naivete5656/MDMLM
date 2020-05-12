function candidate_detection_online_optassociate( im, patch_size, threshold, candidate_sequence_file, ...
                                     first_frame_no, frame_no, delayed_frame_no, param )

% add by ryoma
im = double(im)/2^12*255;
                                 
% parameter settings
minCellArea = param.minCellArea;                                 
                                 
image_height = size( im, 1 );
image_width = size( im, 2 );

if frame_no == first_frame_no
    lastlast_candi_count = 0;
    last_candi_count = 0;
    candi_count = 0;
else
    load( candidate_sequence_file );
end

lastlast_candi_count = last_candi_count;
last_candi_count = candi_count;

patch_size_half = (patch_size-1)/2;

bg_image = conv2( double( im ), ones(100,1)/100, 'same' );
bg_image = conv2( bg_image, ones(1,100)/100, 'same' );    

bg_image(1:49,1:49) = bg_image(50,50);
bg_image(1:49,end-49:end) = bg_image(50,end-50);
bg_image(1:end-49,1:49) = bg_image(end-50,50);
bg_image(end-49:end,end-49:end) = bg_image(end-50,end-50);
bg_image(1:49,50:end-50) = repmat( bg_image(50,50:end-50), [49 1] );
bg_image(end-49:end,50:end-50) = repmat( bg_image(end-50,50:end-50), [50 1] );
bg_image(50:end-50,1:49) = repmat( bg_image(50:end-50,50), [1 49] );
bg_image(50:end-50,end-49:end) = repmat( bg_image(50:end-50,end-50), [1 50] );

img_diff = max( double( im ) - bg_image, 0 );

matconv = conv2( img_diff, ones( patch_size ), 'valid' ) / patch_size^2;
    
bw = bwlabel( double( matconv > threshold ), 4 );
stats = regionprops( bw, 'BoundingBox' );      

statsA = regionprops(bw,'Area');
AreaList = [];
ccc = 0;

for j = 1 : size( stats, 1 )
    AreaList(end+1) = statsA(j).Area;
    if statsA(j).Area < minCellArea;
        ccc = ccc + 1;
        continue;
    end
        
    bound_left = stats(j).BoundingBox(1);
    bound_top = stats(j).BoundingBox(2);
    bound_right = stats(j).BoundingBox(1) + stats(j).BoundingBox(3) - 1 + patch_size/2;
    bound_bottom = stats(j).BoundingBox(2) + stats(j).BoundingBox(4) - 1 + patch_size/2;

    centerx = round((bound_left + bound_right) / 2);
    centery = round((bound_top + bound_bottom) / 2);
    if centerx - patch_size_half <= 0 || centerx + patch_size_half > image_width || ...
       centery - patch_size_half <= 0 || centery + patch_size_half > image_height 
        continue;
    end                

    match = false;
    for k = last_candi_count + 1 : candi_count
        if next(k) == -1 continue; end
        if ( candi{k}(2) <= bound_left && bound_left <= candi{k}(4) || ...
             bound_left <= candi{k}(2) && candi{k}(2) <= bound_right ) && ...
           ( candi{k}(3) <= bound_top && bound_top <= candi{k}(5) || ...
             bound_top <= candi{k}(3) && candi{k}(3) <= bound_bottom )
                sizej = stats(j).BoundingBox(3)* stats(j).BoundingBox(4);
                sizek = (candi{k}(4)-candi{k}(2))*(candi{k}(5)-candi{k}(3));
                if sizej < 600 || sizek < 600;
                   candi{k}(2) = min( candi{k}(2), bound_left );
                   candi{k}(3) = min( candi{k}(3), bound_top );
                   candi{k}(4) = max( candi{k}(4), bound_right );
                   candi{k}(5) = max( candi{k}(5), bound_bottom );
                   match = true;
                   k = candi_count;
                end
        end
    end

    if ~match
        candi_count = candi_count + 1;
        next( candi_count ) = 0;            
        candi{ candi_count } = [ frame_no bound_left bound_top bound_right bound_bottom ];
    end
end

%% optimize association
% make hypotheses
C = []; D=[];
for k1 = lastlast_candi_count + 1 : last_candi_count
    if next(k1) == -1 continue; end
    for k2 = last_candi_count + 1 : candi_count
        if ( candi{k1}(2) <= candi{k2}(2) && candi{k2}(2) <= candi{k1}(4) + patch_size/2 || ...
             candi{k2}(2) <= candi{k1}(2) && candi{k1}(2) <= candi{k2}(4) + patch_size/2 ) && ...
           ( candi{k1}(3) <= candi{k2}(3) && candi{k2}(3) <= candi{k1}(5) + patch_size/2 || ...
             candi{k2}(3) <= candi{k1}(3) && candi{k1}(3) <= candi{k2}(5) + patch_size/2 )   
            % add hypotheses
            C(end+1,:) = zeros(1,candi_count-lastlast_candi_count);
            C(end,k1-lastlast_candi_count) = 1;
            C(end,k2-lastlast_candi_count) = 1;
            hhh= min(candi{k2}(4) - candi{k1}(2), candi{k1}(4) - candi{k2}(2));
            www = min(candi{k2}(5) - candi{k1}(3), candi{k1}(5) - candi{k2}(3));
            D(end+1) = hhh*www;
        end
    end
end

% optimize hypotheses
if length(D)>0
    b = ones(size(C,2),1);
    intcon = [1:length(D)];
    lb = zeros(size(D));
    ub = ones(size(D));
    [x,fval,exitflag,output] = intlinprog(-D,intcon,C',b,[],[],lb,ub);
    x = round(x);
else
    x = [];
end
    
% upload the association
for ii=1:length(x);
    if x(ii)==1;
        [r c] = find(C(ii,:)==1);
        k1 = c(1) + lastlast_candi_count;
        k2 = c(2) + lastlast_candi_count;
        next(k1) = k2; 
    end
end

visited = zeros( candi_count, 1 );
frame_count = [];
seq_count = 0;
seq_id = 0;
seq = [];
seq_no = [];
for i = 1 : candi_count
    if next(i) == -1 || next(i) == 0 || visited(i) continue; end
    seq_count = seq_count + 1;
    seq_id = seq_id + 1;
    frame_count(seq_count) = 1;    
    seq{seq_count}(1) = i;
    visited(i) = i;
    j = i;
    unified = false;
    while(next(j) ~= 0)
        j = next(j);
        visited(j) = i;
        frame_count(seq_count) = frame_count(seq_count) + 1;    
        seq{seq_count}(frame_count(seq_count)) = j;
    end
    included = false;
    for j = 1 : frame_count(seq_count)
        if candi{seq{seq_count}(j)}(1) == frame_no - delayed_frame_no
            included = true;    break
        end
    end
    if included
        seq_no(seq_count) = seq_id;
    else
        seq{seq_count} = [];
        seq_count = seq_count - 1;
    end
end
if exist('candi') == 0;
    candi = [];
end
if exist('next') == 0;
    next = [];
end

save( candidate_sequence_file, 'seq_count', 'seq', 'seq_no', 'frame_count', 'candi_count', 'candi', 'lastlast_candi_count', 'last_candi_count', 'next' );

