package bridge.writer;

public class MongoOperator implements DbOperator{
    @Override
    public void insert(Object obj) {
        System.out.println(obj + "已写入MongoDB");
    }
}
