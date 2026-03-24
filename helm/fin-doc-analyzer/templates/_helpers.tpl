{{- define "fin-doc-analyzer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "fin-doc-analyzer.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "fin-doc-analyzer.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "fin-doc-analyzer.labels" -}}
app.kubernetes.io/name: {{ include "fin-doc-analyzer.name" . }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
