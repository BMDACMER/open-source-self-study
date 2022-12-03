package compose;

import java.util.ArrayList;
import java.util.List;

public class OssDirectory implements OssNode{
    private List<OssNode> subNodes = new ArrayList<>();
    String path = null;

    public OssDirectory(String path) {
        this.path = path;
    }

    public OssDirectory(OssDirectory parent, String dirName) {
        parent.addChild(this);
        this.path = parent.getPath() + dirName;
    }

    public void addChild(OssNode node) {
        this.subNodes.add(node);
    }

    public void removeChild(OssNode node) {
        this.subNodes.remove(node);
    }

    public List<OssNode> getChildren() {
        return this.subNodes;
    }

    @Override
    public String getPath() {
        return this.path;
    }

    @Override
    public String getType() {
        return "directory";
    }

}
