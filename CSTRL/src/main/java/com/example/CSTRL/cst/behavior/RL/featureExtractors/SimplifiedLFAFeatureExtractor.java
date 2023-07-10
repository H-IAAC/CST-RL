package com.example.CSTRL.cst.behavior.RL.featureExtractors;

import java.util.ArrayList;
import java.util.Collections;

public class SimplifiedLFAFeatureExtractor extends FeatureExtractor {
    /*
        A feature extractor that considers a simplified version of the experiment. Assumes that the agent only has
        control over the forwards and backwards motion. As consequence, doesn't consider rotation and x position
        important. Moreover, the prox_idx, instead of considering the effect of every single proximity index, is defined
        as 0 if no car is nearby, 1 if a car is in the upper left corner and 2 if a car is in the lower left corner. It
        also includes a multiplication of the prox_idx by the action_id to account for dependencies between these two

        As such, our feature vector will be [y, prox_idx, action_id, prox_idx * action_id]

        NOTE: Trying [y, prox_idx, a_i, a_i * prox_idx]
     */

    final int proxVectorAmount;

    final double maxY = 576;

    public SimplifiedLFAFeatureExtractor(int proxVectorAmount) {
        this.proxVectorAmount = proxVectorAmount;
    }

    @Override
    public ArrayList<Double> extractFeatures(ArrayList<Double> S) {
        ArrayList<Double> normalizedS = normalizeValues(S);
        ArrayList<Double> features = new ArrayList<>();

        // Coordinates
        features.add(normalizedS.get(1));

        // Proximity index
        double proxIdx = getProxIdx(normalizedS);

        features.add(proxIdx);

        // Action ID
        //double actionIdx = 0;
        //double maxActionValue = S.get(3 + proxVectorAmount);

        //for (int i = 4 + proxVectorAmount; i < S.size(); i++) {
        //    if (S.get(i) > maxActionValue) {
        //        maxActionValue = S.get(i);
        //        actionIdx = i - 3 - proxVectorAmount;
        //    }
        //}

        //features.add(actionIdx);

        // Action ID * Prox idx
        //features.add(actionIdx * proxIdx);

        for (int i = 3 + proxVectorAmount; i < normalizedS.size(); i++) {
            features.add(normalizedS.get(i));
            features.add(normalizedS.get(i) * proxIdx);
        }

        return features;
    }

    private double getProxIdx(ArrayList<Double> S) {
        Double minProx = Double.POSITIVE_INFINITY;
        double proxIdx = 0;

        for (int i = 3; i < 3 + proxVectorAmount; i++) {
            if (S.get(i) < minProx) {
                minProx = S.get(i);
                proxIdx = i - 2;
            }
        }

        if (proxIdx > 3 * proxVectorAmount / 4.0) {
            proxIdx = 0.5;
        } else if (proxIdx > proxVectorAmount / 2.0) {
            proxIdx = 1;
        } else {
            proxIdx = 0;
        }

        return proxIdx;
    }

    @Override
    public int getFeatureVectorSize(int stateSize) {
        return 2 + 2 * (stateSize - 3 - proxVectorAmount);
    }

    @Override
    public ArrayList<ArrayList<Double>> getActionJacobian(ArrayList<Double> S, ArrayList<Double> A) {
        ArrayList<Double> combinedS = new ArrayList<>(S);
        combinedS.addAll(A);

        ArrayList<Double> normalizedS = normalizeValues(combinedS);

        ArrayList<ArrayList<Double>> actionJacobian = new ArrayList<>();
        for (int i = 0; i < A.size(); i++) {
            actionJacobian.add(new ArrayList<>(Collections.nCopies(getFeatureVectorSize(normalizedS.size()), 0.0)));
        }

        double proxIdx = getProxIdx(normalizedS);
        for (int i = 0; i < A.size(); i++) {
            actionJacobian.get(i).set(2 + 2 * i, 1.0);
            actionJacobian.get(i).set(2 + 2 * i + 1, proxIdx);
        }

        return actionJacobian;
    }
}
