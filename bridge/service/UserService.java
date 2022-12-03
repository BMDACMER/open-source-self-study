package bridge.service;

import bridge.logger.Logger;
import bridge.writer.DbOperator;

public class UserService implements Service{
    private DbOperator dbOperator = null;
    private Logger logger = null;

    public UserService(DbOperator dbOperator, Logger logger) {
        this.dbOperator = dbOperator;
        this.logger = logger;
    }

    public void  create() {
        dbOperator.insert("{用户A数据}");
        logger.debug("数据插入成功");
    }
    @Override
    public void init() {
        logger.info("UserService已初始化完毕");
    }
}
