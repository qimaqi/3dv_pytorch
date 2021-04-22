
datasetDir = 'H:/nyu_v2';
sceneName = 'home_offices'; %'bedrooms_part1/bedroom_0001'
sceneName_total = 'home_offices';
targetDir='H:/nyu/nyu_v2_imgs';
%sceneDir =[datasetDir,'/',sceneName];
sceneDir =[datasetDir,'/',sceneName_total]

scene_list = cellstr(ls(sceneDir));
scene_list = sort(scene_list)
n = 0;
for num = 3:numel(scene_list)
    name = char(scene_list(num));
    %disp(name);
    path = [datasetDir,'/',sceneName_total,'/',name];
    %disp(path)
    %fname = cellstr(ls(path))
    frameList = get_synched_frames(path);
    mkdir([targetDir,'/',sceneName,'/rgb']);
    mkdir([targetDir,'/',sceneName,'/depth']);
    for ii = 1:numel(frameList)
        if mod(ii,50) == 0
            imgRgb = imread([path,'/', frameList(ii).rawRgbFilename]);
            n=n+1;
            name=num2str(n);
            imwrite(imgRgb,[targetDir,'/',sceneName,'/rgb/',name, '.jpg']);
        end
        pause(0.01);
    end
end



