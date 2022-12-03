package bridge.writer;

public class MysqlOperator implements DbOperator{
    @Override
    public void insert(Object obj) {
        System.out.println(obj + "已写入Redis");
    }
}
