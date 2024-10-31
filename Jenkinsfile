pipeline {
    agent any 

    stages {
        stage('Stage 1') {
            steps {
                echo 'This is Stage 1'
            }
        }
        stage('Stage 2') {
            steps {
                echo 'This is Stage 2'
            }
        }
        stage('SonarQube Analysis') {
        def scannerHome = tool 'SonarScanner';
        withSonarQubeEnv() {
          sh "${scannerHome}/bin/sonar-scanner"
        }
      }
        stage('Stage 4') {
            steps {
                echo 'This is Stage 4'
            }
        }
    }
}
