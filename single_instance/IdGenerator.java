package single_instance;

import java.util.concurrent.atomic.AtomicLong;

public enum IdGenerator {
    INSTANCE;
    private AtomicLong id = new AtomicLong(0);
    public long getId() {
        return id.incrementAndGet();
    }

    public static void main(String[] args) {
        System.out.println(IdGenerator.INSTANCE.getId());
        System.out.println(IdGenerator.INSTANCE.getId());
        System.out.println(IdGenerator.INSTANCE.getId());
    }
}
