pipeline {
    agent any
    stages {
        stage('Checkout Code') {
            steps {
                git url: 'https://github.com/MohamedAmynElboujadaini/Fast_api_stage.git', branch: 'main'
            }
        }
        stage('Build Docker Image') {
            steps {
                catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                    bat 'docker build -t fast_api .'
                }
            }
        }
    }
}
