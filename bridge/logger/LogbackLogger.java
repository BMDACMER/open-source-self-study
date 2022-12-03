package bridge.logger;

public class LogbackLogger implements Logger{
    @Override
    public void info(String message) {
        System.out.println("logback->[INFO]" + message);
    }

    @Override
    public void debug(String debug) {
        System.out.println("logback->[debug]" + debug);
    }
}
