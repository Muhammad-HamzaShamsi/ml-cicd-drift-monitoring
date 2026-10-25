// Declarative pipeline: test → drift check → (retrain if drift) → build → deploy
pipeline {
    agent any

    triggers {
        cron('H 6 * * *')          // daily drift check
        pollSCM('H/15 * * * *')    // and on new commits
    }

    options {
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timeout(time: 30, unit: 'MINUTES')
    }

    environment {
        VENV            = "${WORKSPACE}/.venv"
        IMAGE           = "drift-monitor"
        PUSHGATEWAY_URL = "localhost:9091"
    }

    stages {
        stage('Setup') {
            steps {
                sh '''
                    python3 -m venv "$VENV"
                    . "$VENV/bin/activate"
                    pip install -q -r requirements-dev.txt
                '''
            }
        }

        stage('Lint & Test') {
            steps {
                sh '''
                    . "$VENV/bin/activate"
                    ruff check .
                    pytest -q --junitxml=reports/junit.xml
                '''
            }
            post {
                always { junit 'reports/junit.xml' }
            }
        }

        stage('Drift Check') {
            steps {
                script {
                    sh '. "$VENV/bin/activate" && python -m drift.simulate --out data --drift ${DRIFT_LEVEL:-0}'
                    def rc = sh(
                        returnStatus: true,
                        script: '. "$VENV/bin/activate" && python -m drift.detect --reference data/reference.csv --current data/current.csv --out reports'
                    )
                    env.DRIFT_DETECTED = (rc == 2) ? 'true' : 'false'
                    if (rc != 0 && rc != 2) { error("Drift check failed with exit code ${rc}") }
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'reports/drift_report.*', allowEmptyArchive: true
                }
            }
        }

        stage('Retrain') {
            when { environment name: 'DRIFT_DETECTED', value: 'true' }
            steps {
                echo 'Drift detected → triggering retraining job'
                // e.g. build job: 'mlops-end-to-end-pipeline/main', wait: true
            }
        }

        stage('Build Image') {
            when { branch 'main' }
            steps {
                sh 'docker build -t ${IMAGE}:${BUILD_NUMBER} -t ${IMAGE}:latest .'
            }
        }
    }

    post {
        failure { echo 'Pipeline failed: check the stage logs above.' }
        success { echo "Pipeline OK. Drift detected: ${env.DRIFT_DETECTED}" }
    }
}
