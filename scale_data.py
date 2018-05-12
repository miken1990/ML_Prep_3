import pandas as pd
import Consts
from ElectionsDataPreperation import ElectionsDataPreperation as EDP

class ScaleData:
    scale_args = dict()

    def scale_train(self, df: pd.DataFrame) -> None:
        scaled = "" # ""_scaled"
        for feature in df.keys():

            if feature == Consts.INDEX_COL:
                continue

            t = df[feature].describe().transpose()
            if feature in Consts.setGaussianFeatures:
                miu, sigma = t["mean"], t["std"]
                df[feature + scaled] = (df[feature] - miu) / sigma
                self.scale_args[feature] = (miu, sigma)

            elif feature in Consts.setUniformFeatures:
                min_val, max_val = t["min"], t["max"]
                df[feature + scaled] = (df[feature] - min_val) * 2 / (max_val - min_val) - 1
                self.scale_args[feature] = (min_val, max_val)
        # df = df.drop(Consts.setUniformFeatures.union(Consts.setGaussianFeatures), axis=1)

    def scale_test(self, df: pd.DataFrame) -> None:
        scaled = ""   # "_scaled"
        for feature in df.keys():

            if feature == Consts.INDEX_COL:
                continue

            if feature in Consts.setGaussianFeatures:
                miu, sigma = self.scale_args[feature]
                df[feature + scaled] = (df[feature] - miu) / sigma

            elif feature in Consts.setUniformFeatures:
                min_val, max_val = self.scale_args[feature]
                df[feature + scaled] = (df[feature] - min_val) * 2 / (max_val - min_val) - 1

        # df = df.drop(Consts.setUniformFeatures.union(Consts.setGaussianFeatures), axis=1)
    def scale_all(self, edp: EDP) -> None:
        self.scale_train(edp.trainData)
        self.scale_test(edp.valData)
        self.scale_test(edp.testData)