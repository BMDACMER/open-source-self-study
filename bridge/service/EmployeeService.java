package bridge.service;

import bridge.logger.Logger;
import bridge.writer.DbOperator;

public class EmployeeService implements Service{
    private DbOperator dbOperator = null;
    private Logger logger = null;

    public EmployeeService(DbOperator dbOperator, Logger logger) {
        this.dbOperator = dbOperator;
        this.logger = logger;
    }

    public void update() {
        dbOperator.insert("{员工A数据}");
        logger.info("数据更新成功");
    }

    @Override
    public void init() {
        logger.info("EmployeeService已初始化完毕");
    }
}
