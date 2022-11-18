package builder;

public class App {
    public static void main(String[] args) {
        RabbitMQClient instance = new RabbitMQClient.Builder().setHost("125").setMode(3).setExchange("guohao").build();
        instance.sendMessage("Test");
        instance.sendMessage("ABC");
    }
}
