
import ClipPreProcessor
import ClipSearch

class OnePick:
    def __init__(self, user_id):
        self.user_id  = user_id

    def preproces_and_save(self):
        cp = ClipPreProcessor.ClipPreProcessor(self.user_id)
        image_features, image_paths = cp.preprocess_images()
        cp.store_image_features(image_features, image_paths)
        return "done"

    def getImagePath(self, query):
        cs = ClipSearch.ClipSearch(query, self.user_id, 5)
        top_list = cs.extract_top_results()
        return top_list