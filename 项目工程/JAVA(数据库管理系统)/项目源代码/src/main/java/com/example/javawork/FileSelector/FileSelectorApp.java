package com.example.javawork.FileSelector;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class FileSelectorApp extends Application {

    @Override
    public void start(Stage primaryStage) throws IOException {

        FXMLLoader loader = new FXMLLoader(getClass().getResource("FileSelector.fxml"));
        Parent root = loader.load();

        // 获取控制器并传递主舞台引用
        FileSelectorController controller = loader.getController();
        controller.setPrimaryStage(primaryStage);

        primaryStage.setTitle("TSV File Table Creator");
        primaryStage.setScene(new Scene(root, 400, 400));
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }

    public void startFileSelector(Stage primaryStage) throws IOException {
        FXMLLoader loader = new FXMLLoader(getClass().getResource("FileSelector.fxml"));
        Parent root = loader.load();

        // 获取控制器并传递主舞台引用
        FileSelectorController controller = loader.getController();
        controller.setPrimaryStage(primaryStage);

        primaryStage.setTitle("TSV File Table Creator");
        primaryStage.setScene(new Scene(root, 400, 400));
        primaryStage.show();
    }

}
