package bridge;

import bridge.logger.Log4jLogger;
import bridge.logger.LogbackLogger;
import bridge.service.EmployeeService;
import bridge.service.UserService;
import bridge.writer.MongoOperator;
import bridge.writer.MysqlOperator;

public class Client {
    public static void main(String[] args) {
        EmployeeService employeeService = new EmployeeService(new MongoOperator(), new Log4jLogger());
        employeeService.init();
        employeeService.update();
        System.out.println("============================");
        UserService userService = new UserService(new MysqlOperator(), new LogbackLogger());
        userService.init();
        userService.create();
    }
}
