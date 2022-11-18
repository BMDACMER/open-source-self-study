package builder;
/**
 * 利用SET方法虽然灵活，但是存在中间状态，且属性校验时有前后顺序约束，或者还需要构建额外的校验方法
 * 并且SET方法破坏了“不可变对象”的密闭性
 */
public class RabbitMQClientSample2 {
    private String host = "127.0.0.1";
    private int port = 5672;
    private int mode;
    private String exchange;
    private String queue;
    private boolean isDurable = true;
    int connectionTimeout = 20;
    //让对象不可变
    private RabbitMQClientSample2(){

    }

    public String getHost() {
        return host;
    }

    public void setHost(String host) {
        this.host = host;
    }

    public int getPort() {
        return port;
    }

    public void setPort(int port) {
        this.port = port;
    }

    public int getMode() {
        return mode;
    }

    public void setMode(int mode) {
        this.mode = mode;

    }

    public String getExchange() {
        return exchange;
    }

    public void setExchange(String exchange) {
        if(mode == 1){ //工作队列模式不需要设置交换机，但queue必填
            if(exchange != null){
                throw new RuntimeException("工作队列模式无须设计交换机");
            }
            if(queue == null || queue.trim().equals("")){
                throw new RuntimeException("工作队列模式必须设置队列名称");
            }
            if(!isDurable){
                throw new RuntimeException("工作队列模式必须开启数据持久化");
            }
        }else if(mode ==2){ //路由模式必须设置交换机，但不能设置queue队列
            if(exchange == null || exchange.trim().equals("")){
                throw new RuntimeException("路由模式请设置交换机");
            }
            if(queue != null){
                throw new RuntimeException("路由模式无须设置队列名称");
            }
        }
        this.exchange = exchange;

    }

    public String getQueue() {
        return queue;
    }

    public void setQueue(String queue) {
        if(mode == 1){ //工作队列模式不需要设置交换机，但queue必填
            if(exchange != null){
                throw new RuntimeException("工作队列模式无须设计交换机");
            }
            if(queue == null || queue.trim().equals("")){
                throw new RuntimeException("工作队列模式必须设置队列名称");
            }
            if(!isDurable){
                throw new RuntimeException("工作队列模式必须开启数据持久化");
            }
        }else if(mode ==2){ //路由模式必须设置交换机，但不能设置queue队列
            if(exchange == null || exchange.trim().equals("")){
                throw new RuntimeException("路由模式请设置交换机");
            }
            if(queue != null){
                throw new RuntimeException("路由模式无须设置队列名称");
            }
        }
        this.queue = queue;

    }

    public boolean isDurable() {
        return isDurable;
    }

    public void setDurable(boolean durable) {
        isDurable = durable;
    }

    public int getConnectionTimeout() {
        return connectionTimeout;
    }

    public void setConnectionTimeout(int connectionTimeout) {
        this.connectionTimeout = connectionTimeout;
    }

    //没办法，必须增加一个额外的validate方法验证对象是否复合要求
    public boolean validate(){
        if(mode == 1){ //工作队列模式不需要设置交换机，但queue必填
            if(exchange != null){
                throw new RuntimeException("工作队列模式无须设计交换机");
            }
            if(queue == null || queue.trim().equals("")){
                throw new RuntimeException("工作队列模式必须设置队列名称");
            }
            if(!isDurable){
                throw new RuntimeException("工作队列模式必须开启数据持久化");
            }
        }else if(mode ==2){ //路由模式必须设置交换机，但不能设置queue队列
            if(exchange == null || exchange.trim().equals("")){
                throw new RuntimeException("路由模式请设置交换机");
            }
            if(queue != null){
                throw new RuntimeException("路由模式无须设置队列名称");
            }
        }
        return true;
        //其他各种验证
    }
    public void sendMessage(String msg) {
        System.out.println("正在发送消息：" + msg);
    }

    public static void main(String[] args) {

        RabbitMQClientSample2 client = new RabbitMQClientSample2();
        client.setHost("192.168.31.210");
        client.setMode(1);
        client.setDurable(true);
        client.setQueue("queue");
        client.validate();
        client.sendMessage("Test");

    }
}
