package com.example.CSTRL.cst.behavior.RL.policies;

import com.example.CSTRL.cst.behavior.RL.RLRates.RLRate;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.FeatureExtractor;
import com.example.CSTRL.cst.behavior.RL.util.RLMath;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;

import java.util.ArrayList;
import java.util.Random;

public class GaussianPolicy extends ContinuousPolicy {
    FeatureExtractor featureExtractor;
    ArrayList<ArrayList<Double>> theta;
    double std;

    public GaussianPolicy(int totalActions, RLRate alpha, FeatureExtractor featureExtractor, int stateSize, double std) {
        super(totalActions, alpha);

        this.featureExtractor = featureExtractor;
        this.std = std;

        theta = new ArrayList<>();
        Random r = new Random();
        for (int i = 0; i < totalActions; i++) {
            theta.add(new ArrayList<>());
            for (int j = 0; j < featureExtractor.getFeatureVectorSize(stateSize); j++) {
                theta.get(i).add(r.nextDouble());
            }
        }
    }

    private ArrayList<Double> getMean(ArrayList<Double> S) {
        ArrayList<Double> m = new ArrayList<>();

        for (int i = 0; i < totalActions; i++) {
            m.add(RLMath.dotProduct(S, theta.get(i)));
        }

        return m;
    }

    @Override
    public void updatePolicy(ArrayList<Double> initialState) {
        // For each step, we update theta by alpha times the gradient of J(theta). Since we're approximating J(theta) as
        // the value for the starting position
    }

    @Override
    public ArrayList<Double> getBestAction(ArrayList<Double> S) {
        return getMean(S);
    }

    @Override
    public ArrayList<Double> getPolicyAction(ArrayList<Double> S, StateActionValueFunction valueFunction) {
        ArrayList<Double> M = getMean(S);

        ArrayList<Double> A = new ArrayList<>();
        Random random = new Random();
        for (int i = 0; i < totalActions; i++) {
            // We use the Box Muller transform to get random variables in the uniform distribution. We are currently
            // getting these distributions independently of each other. As such, even though they all have the correct
            // mean, the resulting distribution won't be exactly gaussian. This is a problem to be solved in the future.
            // For a single action, however, this setup is correct.
            double u1 = random.nextDouble() * 2 - 1;
            double u2 = random.nextDouble() * 2 - 1;

            double x = Math.sqrt(-2 * Math.log(u2)) * Math.cos(2 * Math.PI * u1);

            A.add(RLMath.clamp(M.get(i) + std * x, actionMaximums.get(i), actionMinimums.get(i)));
        }

        return A;
    }

    @Override
    public void endEpisode() {
        alpha.endEpisode();
    }
}
