package com.example.javawork.DataBrowser;


import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class DataBrowserApp extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {

        Parent root = FXMLLoader.load(getClass().getResource("DataBrowser.fxml"));
        primaryStage.setScene(new Scene(root, 1200, 250));
        primaryStage.setTitle("Protein database management system");
        primaryStage.show();

    }

    public static void main(String[] args) {
        launch(args);
    }
}
