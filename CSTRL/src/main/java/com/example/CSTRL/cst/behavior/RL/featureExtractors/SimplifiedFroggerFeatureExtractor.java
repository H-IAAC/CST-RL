package com.example.CSTRL.cst.behavior.RL.featureExtractors;

import java.util.ArrayList;

public class SimplifiedFroggerFeatureExtractor extends TabularFeatureExtractor {
    /*
        A feature extractor that considers a simplified version of the experiment. Assumes that the agent only has
        control over the forwards and backwards motion. As consequence, doesn't consider rotation and x position
        important. Moreover, the prox_idx, instead of considering the effect of every single proximity index, is defined
        as 0 if no car is nearby, 1 if a car is in the upper left corner and 2 if a car is in the lower left corner

        As such, our feature vector will be [y // PY, prox_idx, action_id]
     */

    final int pY;
    final int proxVectorAmount;

    public SimplifiedFroggerFeatureExtractor(int pY, int proxVectorAmount) {
        this.pY = pY;
        this.proxVectorAmount = proxVectorAmount;
    }

    @Override
    public ArrayList<Double> extractFeatures(ArrayList<Double> S) {
        ArrayList<Double> features = new ArrayList<>();

        // Coordinates
        features.add(Math.floor(S.get(1) / pY));

        // Proximity index
        Double minProx = 255.0;
        double proxIdx = 0;

        for (int i = 3; i < 3 + proxVectorAmount; i++) {
            if (S.get(i) < minProx) {
                minProx = S.get(i);
                proxIdx = i - 2;
            }
        }

        if (proxIdx > 3 * proxVectorAmount / 4.0) {
            proxIdx = 1;
        } else if (proxIdx > proxVectorAmount / 2.0) {
            proxIdx = 2;
        } else {
            proxIdx = 0;
        }

        features.add(proxIdx);

        // Action ID
        double actionIdx = 0;
        double maxActionValue = S.get(3 + proxVectorAmount);

        for (int i = 4 + proxVectorAmount; i < S.size(); i++) {
            if (S.get(i) > maxActionValue) {
                maxActionValue = S.get(i);
                actionIdx = i - 3 - proxVectorAmount;
            }
        }

        features.add(actionIdx);

        return features;
    }

    @Override
    public int getFeatureVectorSize(int stateSize) {
        return 3;
    }
}
