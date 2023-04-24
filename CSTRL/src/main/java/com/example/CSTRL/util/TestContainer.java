package com.example.CSTRL.util;

public class TestContainer {
    private final String message;
    private final String addon;
    protected TestType testType = TestType.REGULAR;

    public TestContainer(String message, String addon) {
        this.message = message;
        this.addon = addon;
    }

    public String getMessage() {
        return message;
    }

    public String getAddon() {
        return addon;
    }

    public TestType getTestType() {
        return testType;
    }
}
