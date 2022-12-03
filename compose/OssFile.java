package compose;

public class OssFile implements OssNode{
    private String filename;
    private OssDirectory ossDirectory;

    public OssFile(OssDirectory dir ,String filename) {
        this.ossDirectory = dir;
        this.filename = filename;
        //添加到指定节点
        dir.addChild(this);
    }

    @Override
    public String getPath() {
        return ossDirectory.getPath() + filename;
    }

    @Override
    public String getType() {
        return "file";
    }
}
