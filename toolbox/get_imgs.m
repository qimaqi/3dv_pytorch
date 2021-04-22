
datasetDir = 'H:/nyu_v2';
sceneName = 'home_offices/home_office_0001'
targetDir='H:/nyu/nyu_v2_imgs';
sceneDir =[datasetDir,'/',sceneName];
ls(sceneDir)

frameList = get_synched_frames(sceneDir);
n=0;
mkdir([targetDir,'/',sceneName,'/rgb']);
mkdir([targetDir,'/',sceneName,'/depth']);
for ii = 1:numel(frameList)
    if mod(ii,50) == 0
        imgRgb = imread([sceneDir,'/', frameList(ii).rawRgbFilename]);
        n=n+1;
        name=num2str(n);
        imwrite(imgRgb,[targetDir,'/',sceneName,'/rgb/',name, '.jpg']);
    end
    pause(0.01);
end


