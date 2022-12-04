import numpy as np

def DeltaDeltaP(X, split_val, Y, TREAT):
    left_y_w_t =  [y for y,t,split in zip(Y,TREAT,X) if split<=split_val and t==1]
    left_y_n_t =  [y for y,t,split in zip(Y,TREAT,X) if split<=split_val and t==0]
    rigth_y_w_t = [y for y,t,split in zip(Y,TREAT,X) if split> split_val and t==1]
    rigth_y_n_t = [y for y,t,split in zip(Y,TREAT,X) if split> split_val and t==0]
    #
    return  len(left_y_w_t),\
            len(left_y_n_t),\
            len(rigth_y_w_t),\
            len(rigth_y_n_t),\
            ((np.mean(left_y_w_t) - np.mean(left_y_n_t)) -
                  (np.mean(rigth_y_w_t) - np.mean(rigth_y_n_t)))

def ate(y: np.ndarray, t: np.ndarray):
    tr = y[t==1]
    ct = y[t==0]
    return (np.mean(tr)-np.mean(ct))

class Node:
    def __init__(self, predicted_score):
        self.predicted_score = predicted_score
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class UpliftTreeRegressor:
    def __init__(
            self,
            
            max_depth: int = 3, # максимальная глубина дерева.
            min_samples_leaf: int = 1000, # минимальное необходимое число обучающих объектов в листе дерева.
            min_samples_leaf_treated: int = 300, # минимальное необходимое число обучающих объектов с T=1 в листе дерева.
            min_samples_leaf_control: int = 300, # минимальное необходимое число обучающих объектов с T=0 в листе дерева.
            ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        self.predicted_score = 0
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

    def fit(
            self,
            X: np.ndarray, # массив (n * k) с признаками.
            treatment: np.ndarray, # массив (n) с флагом воздействия.
            y: np.ndarray # массив (n) с целевой переменной.
            ):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y, treatment)
        
    def check_conditions(
            self,
            depth
            ):
        if self.max_depth != None:
            if self.max_depth< depth:
                return False
        return True
    
    def DeltaDelta(self, X, split_val, y, TREAT):
        indices_left = X < split_val
        left_y =  y[indices_left]
        left_t =  TREAT[indices_left]
        rigth_y = y[~indices_left]
        rigth_t = TREAT[~indices_left]
        indicates_treat_left = left_t ==1
        l_w_t =left_y[indicates_treat_left]
        l_n_t =left_y[~indicates_treat_left]
        indicates_treat_right = rigth_t ==1
        r_w_t =rigth_y[indicates_treat_right]
        r_n_t =rigth_y[~indicates_treat_right]
        
        
        
        if l_w_t.shape[0]>=self.min_samples_leaf_treated and\
           l_n_t.shape[0]>=self.min_samples_leaf_control and\
           r_w_t.shape[0]>=self.min_samples_leaf_treated and\
           r_n_t.shape[0]>=self.min_samples_leaf_control and\
           (l_w_t.shape[0]+l_n_t.shape[0])>=self.min_samples_leaf and\
           (r_w_t.shape[0]+r_n_t.shape[0])>=self.min_samples_leaf:
            return  l_w_t.shape[0],\
                    l_n_t.shape[0],\
                    r_w_t.shape[0],\
                    r_n_t.shape[0],\
                    ((np.mean(l_w_t) - np.mean(l_n_t)) -
                          (np.mean(r_w_t) - np.mean(r_n_t)))
        else:
            return None,None,None,None,0
    
    def find_percetiles(self, column_values):
        # column_values - одномерный массив со значениями признака в текущей вершине.
        unique_values = np.unique(column_values)
        if len(unique_values) > 10:
            percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])
        threshold_options = np.unique(percentiles)
        # threshold_options - получившиеся варианты порога. Их и нужно будет перебрать при подборе оптимального порога.
        return threshold_options

    def predict(self, X: np.ndarray):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y, treatment):

        best_idx, best_thr, best_delta = None, None, 0
        for idx in range(self.n_features_):
            thresholds = self.find_percetiles(X[:, idx])
            for i in thresholds:
                lwt,lnt,rwt,rnt,delta = self.DeltaDelta(X[:,idx],i,y,treatment) #X, split_val, Y, TREAT
                
                if np.abs(delta) > np.abs(best_delta):
                    best_delta = delta
                    best_idx = idx
                    best_thr = i
        return best_idx, best_thr, best_delta

    def _grow_tree(self, X, y, treatment, depth=1):
        idx, thr, delta = self._best_split(X, y, treatment)
        predicted_score = ate(y, treatment)
        node = Node(predicted_score=predicted_score)
        if self.check_conditions(
            depth
            ):
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left, t_left = X[indices_left], y[indices_left], treatment[indices_left]
                X_right, y_right, t_right = X[~indices_left], y[~indices_left],treatment[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, t_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, t_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_score