package com.example.CSTRL.util;

import java.util.ArrayList;

public class TypeTestContainer extends TestContainer {
    private final ArrayList<Double> array;
    private final int anInt;

    public TypeTestContainer(String message, String addon, ArrayList<Double> array, int anInt) {
        super(message, addon);
        this.array = array;
        this.anInt = anInt;

        testType = TestType.TYPE;
    }

    public ArrayList<Double> getArray() {
        return array;
    }

    public int getAnInt() {
        return anInt;
    }
}
