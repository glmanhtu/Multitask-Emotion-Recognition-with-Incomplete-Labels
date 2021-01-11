import collections

Dataset_Info = collections.namedtuple("Dataset_Info", 'data_file categories test_data_file')
class PATH(object):
	def __init__(self, opt=None):
		self.Mixed_EXPR = Dataset_Info(data_file = '/home/mvu/Documents/datasets/mixed/mixed_EXPR_annotations.pkl',
			test_data_file = '',
			categories = ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'])
		self.Mixed_AU = Dataset_Info(data_file= '/home/mvu/Documents/datasets/mixed/mixed_AU_annotations.pkl',
			test_data_file = '',
			categories = ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25'])
		self.Mixed_VA = Dataset_Info(data_file= '/home/mvu/Documents/datasets/mixed/mixed_VA_annotations.pkl',
                        test_data_file = '',
                        categories = ['valence', 'arousal'])
		self.Aff_wild2 = Dataset_Info(data_file = '/home/mvu/Documents/datasets/mixed/affwild-2/annotations.pkl',
			                          test_data_file = '/media/Samsung/Aff-wild2-Challenge/test_set/test_set.pkl',
			categories = {'AU': ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25'],
                            'EXPR':['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'],
                            'VA':['valence', 'arousal']})
		# pytorch benchmark
		self.MODEL_DIR = '/home/mvu/Documents/pretrained'


