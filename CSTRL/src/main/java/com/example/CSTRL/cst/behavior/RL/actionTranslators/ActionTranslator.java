package com.example.CSTRL.cst.behavior.RL.actionTranslators;

import java.util.ArrayList;

public abstract class ActionTranslator {
    public abstract ArrayList<Double> translateAction(ArrayList<Double> A);
}
