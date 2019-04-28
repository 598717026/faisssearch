import numpy as np
import faiss
import pickle
import sys
import cv2
import os
import heapq


INDEX_KEY = "IDMap,IMI2x10,Flat"
TOP_N = 5
SIMILARITY = 1
class index:
    def __init__(self):
        self.imgList = None
        self.detector = None
        self.dimensions = 0
        self.indexPath = "./index"
        self.idsVectorsPath = "./ids_paths_vectors"
        self.index_dict = None
        self.index = None
        self.resizeSize = 0

    def iterate_files(self, _dir):
        result = []
        for root, dirs, files in os.walk(_dir, topdown=True):
            for fl in files:
                if fl.endswith("jpg") or fl.endswith("JPG"):
                    result.append(os.path.join(root, fl))
        return result

    def setResize(self, size):
        self.resizeSize = size

    def setImageList(self, imgList):
        self.imgList = imgList

    def setImageDir(self, imgDir):
        self.imgList = self.iterate_files(imgDir)

    def setFeatureType(self, ft):
        if ft == 'sift':
            self.detector = cv2.xfeatures2d.SIFT_create()
            self.dimensions = 128
        elif ft == 'surf':
            self.detector = cv2.xfeatures2d.SURF_create(800)
            self.dimensions = 64
        elif ft == 'orb':
            self.detector = cv2.ORB_create(1000)
            self.dimensions = 32
        elif ft == 'akaze':
            self.detector = cv2.AKAZE_create()
            self.dimensions = 61
        elif ft == 'brisk':
            self.detector = cv2.BRISK_create()
            self.dimensions = 64

    def getImgaeFeature(self, filename):
        try:
            print(filename)
            img = cv2.imread(filename.encode("utf-8", "surrogateescape").decode(), 0)
            if self.resizeSize > 0:
                if img.shape[0] > img.shape[1]:
                    if img.shape[1] > self.resizeSize:
                        img = cv2.resize(img, (self.resizeSize, int(img.shape[0] * self.resizeSize / img.shape[1])))
                    else:
                        if img.shape[0] > self.resizeSize:
                            img = cv2.resize(img, (int(img.shape[1] * self.resizeSize / img.shape[0]), self.resizeSize))
            kp, desc = self.detector.detectAndCompute(img, None)
        except Exception as e:
            print("getImgaeFeature except", e)
            desc = None
        return desc

    def setIndexPath(self, indexPath):
        self.indexPath = indexPath

    def setIdsVectorsPath(self, idsVectorsPath):
        self.idsVectorsPath = idsVectorsPath

    def makeIndex(self):
        # prepare index
        index = faiss.index_factory(self.dimensions, INDEX_KEY)
        
        ids_count = 0
        index_dict = {}
        ids = None
        features = np.matrix([])
        for file_name in self.imgList:
            print(ids_count)
            feature = self.getImgaeFeature(file_name)
            if feature is None:
                continue
            if feature.any():
                # record id and path
                image_dict = {ids_count: (file_name, feature)}
                index_dict.update(image_dict)
                ids_list = np.linspace(ids_count, ids_count, num=feature.shape[0], dtype="int64")
                ids_count += 1
                if features.any():
                    features = np.vstack((features, feature))
                    ids = np.hstack((ids, ids_list))
                else:
                    features = feature
                    ids = ids_list
                if ids_count % 500 == 499:
                    if not index.is_trained and INDEX_KEY != "IDMap,Flat":
                        print(type(features))
                        index.train(features)
                    index.add_with_ids(features, ids)
                    ids = None
                    features = np.matrix([])

        print(len(features), ids_count)
        if features.any():
            if not index.is_trained and INDEX_KEY != "IDMap,Flat":
                print("train ")
                index.train(features)
            index.add_with_ids(features, ids)

        
        self.index_dict = index_dict
        self.index = index
    
    def saveIndexDict(self):# save index
        faiss.write_index(self.index, self.indexPath)

        # save ids
        with open(self.idsVectorsPath, 'wb') as f:
            try:
                pickle.dump(self.index_dict, f, True)
            except EnvironmentError as e:
                print('Failed to save index file error:[{}]'.format(e))
                f.close()
            except RuntimeError as v:
                print('Failed to save index file error:[{}]'.format(v))
        f.close()



    def loadIndexFromFile(self, index_path = None):
        if index_path is None:
            index_path = self.indexPath
        self.index = faiss.read_index(index_path)


    def loadIdsVectorsFromFile(self, ids_vectors_path = None):
        if ids_vectors_path is None:
            ids_vectors_path = self.idsVectorsPath
        if not os.path.exists(ids_vectors_path):
            return

        with open(ids_vectors_path, 'rb') as f:
            self.index_dict = pickle.load(f)

    def id_to_vector(self, id_):
        try:
            return self.index_dict[id_]
        except:
            pass

    def search_by_image(self, image, k):
        ids = [None]
        vectors = self.getImgaeFeature(image)
        if vectors is None:
            results = [{}]
        else:
            results = self.__search__(ids, [vectors], k)

        return results

    def __search__(self, ids, vectors, topN):
        def neighbor_dict_with_path(id_, file_path, score):
            return {'id': int(id_), 'file_path': file_path, 'score': score}

        def neighbor_dict(id_, score):
            return {'id': int(id_), 'score': score}

        def result_dict_str(id_, neighbors):
            return {'id': id_, 'neighbors': neighbors}

        results = []
        need_hit = SIMILARITY

        for id_, feature in zip(ids, vectors):
            scores, neighbors = self.index.search(feature, k=topN) if feature.size > 0 else ([], [])
            n, d = neighbors.shape
            result_dict = {}

            for i in range(n):
                l = np.unique(neighbors[i]).tolist()
                for r_id in l:
                    if r_id != -1:
                        score = result_dict.get(r_id, 0)
                        score += 1
                        result_dict[r_id] = score

            h = []
            for k in result_dict:
                v = result_dict[k]
                if v >= need_hit:
                    if len(h) < topN:
                        heapq.heappush(h, (v, k))
                    else:
                        heapq.heappushpop(h, (v, k))

            result_list = heapq.nlargest(topN, h, key=lambda x: x[0])
            neighbors_scores = []
            for e in result_list:
                confidence = e[0] * 100 / n
                if self.id_to_vector:
                    file_path = self.id_to_vector(e[1])[0]
                    neighbors_scores.append(neighbor_dict_with_path(e[1], file_path, str(confidence)))
                else:
                    neighbors_scores.append(neighbor_dict(e[1], str(confidence)))
            results.append(result_dict_str(id_, neighbors_scores))
        return results

if __name__ == '__main__':
    import time

    idxc = index()
    idxc.setFeatureType("sift")
    idxc.setImageDir("/data/lhg_work/booksmaker2test/data/books/test/")
    idxc.setIndexPath("./index")
    idxc.setIdsVectorsPath("./ivpath")
    idxc.setResize(480)
    idxc.makeIndex()
    idxc.saveIndexDict()
    idxc.loadIndexFromFile()
    idxc.loadIdsVectorsFromFile()
    t1 = time.time()
    # result = idxc.search_by_image("/data/works/robot/data/pic/2019-02-25-22-03-03_cover_none.jpg", 5)
    t2 = time.time()
    print("search time:", t2 - t1)
    print(result)
