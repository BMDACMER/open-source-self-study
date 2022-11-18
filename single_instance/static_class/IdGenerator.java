package single_instance.static_class;

import java.util.concurrent.atomic.AtomicLong;

public class IdGenerator {
    // 静态内部类写法
    private AtomicLong id = new AtomicLong(0);
    private IdGenerator() {}
    // SingletonHolder 是一个静态内部类，当外部类 IdGenerator 被加载的时候，并不会创建SingletonHolder 实例对象。
    // 只有当调用 getInstance() 方法时，SingletonHolder才会被加载，这个时候才会创建 instance。instance 的唯一性、
    // 创建过程的线程安全性，都由 JVM来保证。所以，这种实现方法既保证了线程安全，又能做到延迟加载。
    private static class SingleHolder{
        private static final IdGenerator instance = new IdGenerator();
    }

    public static IdGenerator getInstance() {
        return SingleHolder.instance;
    }

    public long getId() {
        return id.incrementAndGet();
    }
}
