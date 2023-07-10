package com.example.CSTRL.cst.behavior.RL.featureExtractors;

import java.util.ArrayList;

public class FroggerFeatureExtractor extends TabularFeatureExtractor {
    /*
        On the continuous Frogger application, the state vector is [x, y, rot, proximity data..., action data...].

        We will discretize the values of x, y and rot with a particular precision, and, to minimize the size of our
        Q-table, we will forego the specific proximity data in favor of a single number that indicates the closest car,
        as well as the action components in favor of a single action ID.

        As such, our feature vector will be [x // PX, y // PY, rot // PROT, prox_idx, action_id]
    */

    final int pX;
    final int pY;
    final int pRot;
    final int proxVectorAmount;

    public FroggerFeatureExtractor(int pX, int pY, int pRot, int proxVectorAmount) {
        this.pX = pX;
        this.pY = pY;
        this.pRot = pRot;
        this.proxVectorAmount = proxVectorAmount;
    }

    @Override
    public ArrayList<Double> extractFeatures(ArrayList<Double> S) {
        ArrayList<Double> features = new ArrayList<>();

        // Coordinates
        features.add(Math.floor(S.get(0) / pX));
        features.add(Math.floor(S.get(1) / pY));
        features.add(Math.floor(S.get(2) / pRot));

        // Proximity index
        Double minProx = 255.0;
        double proxIdx = 0;

        for (int i = 3; i < 3 + proxVectorAmount; i++) {
            if (S.get(i) < minProx) {
                minProx = S.get(i);
                proxIdx = i - 2;
            }
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
        return 4 + proxVectorAmount;
    }

    @Override
    public ArrayList<ArrayList<Double>> getActionJacobian(ArrayList<Double> S, ArrayList<Double> A) {
        return null;
    }
}
