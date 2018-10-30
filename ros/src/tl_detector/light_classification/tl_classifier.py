from styx_msgs.msg import TrafficLight
import tensorflow as tf
import label_map_util
import numpy as np
import os, rospy, datetime

# Path to frozen detection graph.
# This is the model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = './light_classification/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('light_classification', 'mscoco_label_map.pbtxt')

# ID value used in coco dataset to represent a traffic light object.
TRAFFIC_LIGHT_ID = 10

class TLClassifier(object):
    def __init__(self):
        # Load a (frozen) Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Loading label map
        category_index = label_map_util.create_category_index_from_labelmap(
            PATH_TO_LABELS, use_display_name=True
        )

        print('tensorflow version: ' + tf.__version__)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        t1 = datetime.datetime.now()

        # Convert image to numpy array
        # image_np = self.load_image_into_numpy_array(image)
        image_np = image

        # Expand dimensions since the model expects images to have shape:
        # [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        output_dict = self.run_inference_for_single_image(image_np,
                                                          self.detection_graph)

        # Iterate through objects
        for i, detection_class in enumerate(output_dict['detection_classes']):
            if detection_class == 10:
                detection_score = output_dict['detection_scores'][i]
                rospy.loginfo("Detected Traffic Light with score "
                              + str(detection_score) + "%")
            if output_dict['detection_scores'][i] == 0.:
                break # Remaining scores are not detections

        t2 = datetime.datetime.now() - t1
        rospy.loginfo("Inference time: " + str(t2))

        #TODO implement light color prediction
        return TrafficLight.UNKNOWN

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self, image, graph):
      with graph.as_default():
        with tf.Session() as sess:
          # Get handles to input and output tensors
          ops = tf.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
          if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict
