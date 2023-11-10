package com.example.CSTRL.cst.behavior;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.cst.behavior.RL.actionSpaces.ActionSpace;
import com.example.CSTRL.cst.behavior.RL.learners.RLLearner;
import com.example.CSTRL.util.RLPercept;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

public class RLCodelet extends Codelet {
    final private int MAX_TRIAL_SIZE = 500;
    final protected RLLearner learner;
    private ActionSpace actionSpace;
    protected ArrayList<RLPercept> trial;

    protected int stepCounter;

    protected double cumulativeReward;
    private final ArrayList<String[]> cumulativeRewardData;
    protected final String info;

    private MemoryObject RLPerceptMO;
    private MemoryObject RLActionMO;

    public RLCodelet(RLLearner learner, ActionSpace actionSpace, MemoryObject perceptMO) {
        this.learner = learner;
        this.actionSpace = actionSpace;
        trial = new ArrayList<>();

        isMemoryObserver = true;
        perceptMO.addMemoryObserver(this);
        
        stepCounter = 0;

        // Graph data
        cumulativeReward = 0;
        cumulativeRewardData = new ArrayList<>();

        info = learner.toString();
    }

    @Override
    public void accessMemoryObjects() {
        RLPerceptMO = (MemoryObject) getInput("RLPERCEPT");
        RLActionMO = (MemoryObject) getOutput("RLACTION");
    }

    @Override
    public void calculateActivation() {

    }

    @Override
    public void proc() {
        learner.setActionSpace(actionSpace);

        // Gets percept
        RLPercept percept = (RLPercept) RLPerceptMO.getI();
        if (percept.getState().size() > 0) { // Will only act if this MO has been updated at least once
            trial.add(percept);
            cumulativeReward += percept.getReward();
            stepCounter++;

            // Updates RL
            if (trial.size() > 1) {
                // The trial given at this point will not have the action for the current state defined
                learner.rlStep(trial);
            }

            // Gets action
            ArrayList<Double> nextAction = learner.selectAction(percept.getState());
            percept.setAction(nextAction);

            RLActionMO.setI(nextAction);

            // Checks for end of episode
            endStep(percept.isTerminal());
        }
    }
    
    // Does any processing that needs to be done at the end of the episode, such as resetting the episode
    protected void endStep(boolean terminal) {
        if (trial.size() > MAX_TRIAL_SIZE) {
            trial.remove(0);
        }
    }

    // Adds a data point to data graph. Can be extended in child classes if they want to generate different graphs. The
    // provided x can be episode count in episodical RL or step counter otherwise
    protected void addGraphDataPoint(String x) {
        cumulativeRewardData.add(new String[] {x, Double.toString(cumulativeReward)});
    }

    // Saves the data graph. Can be extended in child classes if they want to generate differente graphs
    protected void saveGraphData() {
        saveGraph(cumulativeRewardData, "C:\\Users\\morai\\OneDrive\\Documentos\\Git\\CST-RL\\CSTRL\\graphs\\" + info + ".csv");
    }

    protected void saveGraph(ArrayList<String[]> data, String outputPath) {
        // TODO: Make it append to csv instead of saving everything at every episode
        // TODO: Add support for non episodic RL
        File csvOutputFile = new File(outputPath);

        if (!csvOutputFile.exists()) {
            try {
                csvOutputFile.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        try (PrintWriter pw = new PrintWriter(csvOutputFile)) {
            data.stream()
                .map(this::convertToCSV)
                .forEach(pw::println);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private String convertToCSV(String[] data) {
        return String.join(",", data);
    }
}
