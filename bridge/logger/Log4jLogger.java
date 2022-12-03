package bridge.logger;

public class Log4jLogger implements Logger{
    @Override
    public void info(String message) {
        System.out.println("log4j->[INFO]" + message);
    }

    @Override
    public void debug(String debug) {
        System.out.println("log4j->[debug]" + debug);
    }
}
