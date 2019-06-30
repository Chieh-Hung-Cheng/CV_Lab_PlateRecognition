from pathlib import Path

img_dir = Path('./ccpd5000/train/')
img_paths_all = img_dir.glob('*.jpg')
img_paths_list = list(img_paths_all)
img_paths = sorted(img_paths_list) #sorted image paths
#img_paths = img_dir.glob('*.jpg')
#img_paths = sorted(list(img_paths))


print('data size: '+str(len(img_paths)))
print('sample image name: ' + str(img_paths[8]))
#print(len(img_paths))
#print(img_paths[:5])

testname = img_paths[8].name
print(testname)

split_component = testname.split('-')
information = split_component[3]
## = name.split('-')[3]
##print(token)

information_split1 = "&".join(information.split('_'))
information_split2 = information_split1.split('&')
#token = token.replace('&', '_')
#print(token)

#values = token.split('_')
#print(values)

values = list()
for i in range(len(information_split2)):
    values.append(float(information_split2[i]))
print(values)

#values = [float(val) for val in values]
#print(values)