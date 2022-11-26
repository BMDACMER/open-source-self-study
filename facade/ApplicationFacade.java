package facade;

public class ApplicationFacade {
    private CacheManager cacheManager = new CacheManager();
    private DatabaseManager databaseManager = new DatabaseManager();
    private WebServerManager webServerManager = new WebServerManager();
    public void initSystem() {
        databaseManager.startDBCluster();
        cacheManager.initRedis();
        webServerManager.loadWebApp();
    }
}
