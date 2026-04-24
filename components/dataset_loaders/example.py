from dataset_loader import LinkedDataHandler, Annotation


def __main__():
    handler = LinkedDataHandler("/media/jan/data/data")
    print(handler.list_subsets())
    subset = handler.subsets["rosbag2_2025_09_18-16_04_08_out"]
    link = subset['links'][100]
    handler.visualize("rosbag2_2025_09_18-16_04_08_out", link)


    #iterate over all links and print average number of points in the annotations for each link
    #not guaranteed that the links are in order
    for subset_name in handler.list_subsets():
        subset = handler.subsets[subset_name]
        for link in subset["links"]:
            avg = 0
            for annotation in link['samples']['lidar']['annotations']:
                avg += annotation['num_points']
            if len(link['samples']['lidar']['annotations']) > 0:
                print(link['token'], avg / len(link['samples']['lidar']['annotations']))
    
    #iterate over links in order
    for subset_name in handler.list_subsets():
        subset = handler.subsets[subset_name]
        #find first link
        current_link = None
        for link in subset['links']:
            if link['token'] == subset['meta']['first_link_token']:
                current_link = link
                break 
        
        while current_link is not None and current_link['next_link'] is not None:
            #do something here
            current_link = current_link['next_link']
        print("subset: ", subset_name, " last link token: ", current_link['token'])
        


__main__()