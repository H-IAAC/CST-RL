package com.example.CSTRL.cst.behavior.RL.actionTranslators;

import java.util.ArrayList;
import java.util.Collections;

public class SimplifiedActionTranslator extends ActionTranslator {
    @Override
    public ArrayList<Double> translateAction(ArrayList<Double> A) {
        ArrayList<Double> action = new ArrayList<>(Collections.nCopies(4, 0.0));

        double v = A.get(0);
        action.set(0, Math.max(0.0, v));
        action.set(1, -Math.min(0.0, v));

        return action;
    }
}
