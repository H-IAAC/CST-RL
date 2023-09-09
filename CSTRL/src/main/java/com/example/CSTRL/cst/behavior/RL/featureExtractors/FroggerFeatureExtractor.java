package com.example.CSTRL.cst.behavior.RL.featureExtractors;

import java.util.ArrayList;

public class FroggerFeatureExtractor extends TabularFeatureExtractor {
    /*
        On the continuous Frogger application, the state vector is [x, y, rot, proximity data..., action data...].

        We will discretize the values of x, y and rot with a particular precision, and, to minimize the size of our
        Q-table, we will forego the specific proximity data in favor of a single number that indicates the closest car,
        as well as the action components in favor of a single action ID.

        As such, our feature vector will be [x_digital, y_digital, {raycasts_digital}, {action_idx}]
    */

    final int raycastAmount;
    final int raycastResolution;
    final int xResolution;
    final int yResolution;

    public FroggerFeatureExtractor(int raycastAmount, int xResolution, int yResolution, int raycastResolution) {
        this.xResolution = xResolution;
        this.yResolution = yResolution;
        this.raycastResolution = raycastResolution;
        this.raycastAmount = raycastAmount;
    }

    @Override
    public ArrayList<Double> extractFeatures(ArrayList<Double> S) {
        ArrayList<Double> features = new ArrayList<>();

        features.add(Math.floor(S.get(0) / xResolution));
        features.add(Math.floor(S.get(1) / yResolution));

        // Proximity index
        for (int i = 2; i < 2 + raycastAmount; i++) {
            features.add(Math.floor(S.get(i) / raycastResolution));
        }

        // Action ID
        double maxActionValue = 0.0;
        int actionAmount = S.size() - 2 - raycastAmount;
        int actionIdx = actionAmount;

        for (int i = 0; i < actionAmount; i++) {
            if (S.get(2 + raycastAmount + i) > maxActionValue) {
                maxActionValue = S.get(2 + raycastAmount + i);
                actionIdx = i;
            }
        }

        features.add((double) actionIdx);

        return features;
    }

    @Override
    public int getFeatureVectorSize(int stateSize) {
        // x, y, raycasts, action
        return 3 + raycastAmount;
    }

    @Override
    public ArrayList<ArrayList<Double>> getActionJacobian(ArrayList<Double> S, ArrayList<Double> A) {
        return null;
    }
}
