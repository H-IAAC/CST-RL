package com.example.CSTRL.cst.behavior;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.cst.behavior.RL.actionSelectors.ActionSelector;
import com.example.CSTRL.cst.behavior.RL.actionTranslators.ActionTranslator;
import com.example.CSTRL.util.RLPercept;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.Stream;

abstract public class RLCodelet extends Codelet {
    private ActionTranslator actionTranslator;

    protected ArrayList<Double> pastState;
    protected ArrayList<Double> pastAction;

    protected ArrayList<Double> currentState;
    protected Double reward;
    
    protected int stepCounter;

    private MemoryObject RLPerceptMO;
    private MemoryObject RLActionMO;

    protected double cumulativeReward;
    private final ArrayList<String[]> cumulativeRewardData;
    protected final String creationTime;

    public RLCodelet(MemoryObject perceptMO) {
        isMemoryObserver = true;
        perceptMO.addMemoryObserver(this);
        
        stepCounter = 0;

        // Graph data
        cumulativeReward = 0;

        cumulativeRewardData = new ArrayList<>();

        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
        LocalDateTime now = LocalDateTime.now();
        creationTime = dtf.format(now);
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
        RLPercept percept = (RLPercept) RLPerceptMO.getI();

        if (percept.getEnded()) {
            int t = 1;
        }

        currentState = percept.getState();

        if (pastState != null) {
            reward = percept.getReward();
            cumulativeReward += reward;

            runRLStep();
        }

        pastAction = selectAction();

        if (actionTranslator != null) {
            RLActionMO.setI(actionTranslator.translateAction(pastAction));
        } else {
            RLActionMO.setI(pastAction);
        }

        pastState = currentState;
        stepCounter++;

        endStep(percept.getEnded());
    }

    public void setActionTranslator(ActionTranslator actionTranslator) {
        this.actionTranslator = actionTranslator;
    }

    // Update step for the RL algorithm
    abstract protected void runRLStep();

    // Returns the action that should be taken in this step
    abstract protected ArrayList<Double> selectAction();
    
    // Does any processing that needs to be done at the end of the episode, such as resetting the episode
    abstract protected void endStep(boolean episodeEnded);

    // Adds a data point to data graph. Can be extended in child classes if they want to generate different graphs. The
    // provided x can be episode count in episodical RL or step counter otherwise
    protected void addGraphDataPoint(String x) {
        cumulativeRewardData.add(new String[] {x, Double.toString(cumulativeReward)});
    }

    // Saves the data graph. Can be extended in child classes if they want to generate differente graphs
    protected void saveGraphData() {
        saveGraph(cumulativeRewardData, "C:\\Users\\morai\\OneDrive\\Documentos\\Git\\CST-RL\\CSTRL\\graphs\\cumulativeRewardData-" + creationTime + ".csv");
    }

    protected void saveGraph(ArrayList<String[]> data, String outputPath) {
        File csvOutputFile = new File(outputPath);

        if (!csvOutputFile.exists()) {
            try {
                csvOutputFile.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        boolean test = csvOutputFile.exists();

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
