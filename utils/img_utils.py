import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)



def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   



def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    



def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()



def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


def show_bboxes(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    img_height, img_width = sorted_anns[0]['segmentation'].shape
    
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    img = np.ones((img_height, img_width, 4))
    img[:, :, 3] = 0  # Set the alpha channel to be fully transparent
    for ann in sorted_anns:
        # Extract the bounding box in XYWH format
        x_min, y_min, width, height = ann['bbox']
        x_min, y_min, width, height = map(int, [x_min, y_min, width, height])
        
        # x_max와 y_max 계산
        x_max = x_min + width
        y_max = y_min + height
        
        # Generate a random color for the bounding box
        color = np.random.random(3)
        
        # Draw the bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color.tolist() + [0.5], thickness=5)
        
        stability_score_text = f"{ann['stability_score']:.2f}"
        text_position = (x_min, y_min - 10)
        cv2.putText(img, stability_score_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=color.tolist() + [1.0], thickness=1)
        
        if borders:
            # Optional: Draw borders or additional contours if needed
            # Here we will simply add a border around the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0, 0.8), thickness=1)
    
    ax.imshow(img)