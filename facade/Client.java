package facade;

public class Client {
    public static void main(String[] args) {
//        new DatabaseManager().startDBCluster();
//        new CacheManager().initRedis();
//        new WebServerManager().loadWebApp();
        // 下面方法更好 相当于对上面的方法进一步包装
        ApplicationFacade application = new ApplicationFacade();
        application.initSystem();
    }
}
