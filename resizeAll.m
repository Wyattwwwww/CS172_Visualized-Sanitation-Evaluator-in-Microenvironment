maindir = '256_ObjectCategories';
subdirs  = dir(maindir);

for i = 1 : size(subdirs,1)
    if( isequal(subdirs(i).name, '.')||...
        isequal(subdirs(i).name, '..')||...
        ~subdirs(i).isdir)           
        continue;
    end
    subdirpath = fullfile(maindir, subdirs(i).name, '*.jpg');
    imgs = dir(subdirpath);         

    for j = 1 : size(imgs,1)
        imgpath = fullfile(maindir, subdirs(i).name, imgs(j).name);
        img = imread(imgpath);
        l = size(img, 1);
        w = size(img, 2);
        if l > w
            img = imresize(img, [250 NaN]);
        else
            img = imresize(img, [NaN 250]);
        end
        
        new_path = fullfile('256_ResizedTo250', subdirs(i).name);
        if  exist(fullfile(new_path),'dir')==0
            mkdir(new_path);
        end
        new_file = fullfile(new_path, imgs(j).name);
        imwrite(img, new_file);
    end
end