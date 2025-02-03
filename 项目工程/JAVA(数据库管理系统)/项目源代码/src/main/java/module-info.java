module com.example.javawork {
    requires javafx.controls;
    requires javafx.fxml;
    requires java.sql;


    opens com.example.javawork.FileSelector to javafx.fxml;
    opens com.example.javawork.DataBrowser to javafx.fxml;
    opens com.example.javawork.ShowData to javafx.fxml;
    opens com.example.javawork.Search to javafx.fxml;
    opens com.example.javawork.ShowSearchData to javafx.fxml;
    exports com.example.javawork.Database;
    exports com.example.javawork.Parser;
    exports com.example.javawork.FileSelector;
    exports com.example.javawork.DataBrowser;
    exports com.example.javawork.ShowData;
    exports com.example.javawork.Search;
    exports com.example.javawork.ShowSearchData;
}