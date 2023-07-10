package com.example.CSTRL.cst.behavior.RL.util;

import java.util.ArrayList;

public class RLMath {
    public static Double dotProduct(ArrayList<Double> a1, ArrayList<Double> a2) {
        if (a1.size() != a2.size()) {
            throw new RuntimeException("Arrays must be of same size");
        }

        double sum = 0.0;
        for (int i = 0; i < a1.size(); i++) {
            sum += a1.get(i) * a2.get(i);
        }

        return sum;
    }

    public static Double clamp(Double v, Double max, Double min) {
        return Math.max(min, Math.min(max, v));
    }
}
