package com.example.CSTRL.cst;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.cst.behavior.EpisodicRLCodelet;
import com.example.CSTRL.cst.behavior.RL.actionSpaces.ActionSpace;
import com.example.CSTRL.cst.behavior.RL.actionSpaces.DiscreteActionSpace;
import com.example.CSTRL.cst.behavior.RL.learners.TensorforceLearner;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TensorforceAgentMind extends AgentMind {
    final String configPath = "C:\\Users\\morai\\OneDrive\\Documentos\\Git\\CST-RL\\CSTRL\\src\\main\\java\\com\\example\\CSTRL\\cst\\behavior\\RL\\configs\\tensorforce.json";
    final String APIUrl = "http://127.0.0.1:5000";

    @Override
    protected Codelet getRLCodelet(MemoryObject perceptMO) {

        ArrayList<ArrayList<Double>> actions = new ArrayList<ArrayList<Double>>() {
            {
                add(new ArrayList<Double>(Arrays.asList(0.0, 0.0, 0.0, 0.0)));
                add(new ArrayList<Double>(Arrays.asList(1.0, 0.0, 0.0, 0.0)));
                add(new ArrayList<Double>(Arrays.asList(0.0, 1.0, 0.0, 0.0)));
                add(new ArrayList<Double>(Arrays.asList(0.0, 0.0, 1.0, 0.0)));
                add(new ArrayList<Double>(Arrays.asList(0.0, 0.0, 0.0, 1.0)));
            }
        };

        /*
        ArrayList<ArrayList<Double>> actions = new ArrayList<ArrayList<Double>>() {
            {
                add(new ArrayList<Double>(List.of(0.0)));
                add(new ArrayList<Double>(List.of(1.0)));
            }
        };
        */
        ActionSpace actionSpace = new DiscreteActionSpace(actions);

        TensorforceLearner tensorforceLearner;
        try {
            tensorforceLearner = new TensorforceLearner(configPath, APIUrl);
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }

        return new EpisodicRLCodelet(tensorforceLearner, actionSpace, perceptMO);
    }
}
